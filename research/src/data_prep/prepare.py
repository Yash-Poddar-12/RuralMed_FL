"""One CLI for the full teammate workflow and the offline local dev sample."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from . import adapters
from .acquire import acquire, disk_preflight, sha256, extract_verified, verified_extraction
from .common import CLASSES, REPO_ROOT, config_hash, load_config, write_json
from .preprocess import run as preprocess
from .split import run as split
from research.src.partitioning.partition import run as partition


def verify_processed(config):
    root = Path(config["paths"]["processed"])
    manifest = pd.read_csv(root / "manifest.csv", keep_default_na=False)
    for row in manifest.itertuples():
        path = (root / row.processed_path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Processed path escaped data root")
        with Image.open(path) as image:
            import hashlib
            image.load()
            if image.mode != "L" or image.size != (224, 224) or hashlib.sha256(image.tobytes()).hexdigest() != row.pixel_sha256:
                raise ValueError(f"Dev image failed verification: {path}")
    return manifest


def dev_sample(config, source_root=None):
    """Sample existing inputs only. Never call any network/acquisition function."""
    base = Path(config["paths"]["processed"]).parent
    completed = base / "sample.json"
    if completed.exists():
        saved = json.loads(completed.read_text(encoding="utf-8"))
        if saved["config_hash"] != config_hash(config):
            raise ValueError("Dev configuration changed; choose a fresh dev sample")
        verify_processed(config)
        split(config)
        partition(config)
        print(f"Verified existing dev sample: {saved['images']} images", flush=True)
        return saved
    full = load_config(REPO_ROOT / "research/configs/phase1.yml")
    raw = Path(source_root or full["paths"]["raw"])
    rows = []
    missing = []
    for source in config["datasets"]:
        folder = raw / source
        if source == "nih" and not (folder / "extracted").exists():
            folder = raw / "nih-representative"
        if source == "chexpert":
            candidates = [folder, folder / "CheXpert-v1.0-small", folder / "extracted/CheXpert-v1.0-small", folder / "extracted"]
            folder = next((p for p in candidates if (p / "train.csv").is_file()), folder)
            if not (folder / "train.csv").is_file():
                missing.append(source)
                continue
            iterator = adapters.chexpert(folder)
        elif (folder / "extracted").exists():
            iterator = getattr(adapters, source)(folder / "extracted")
        else:
            missing.append(source)
            continue
        rows.extend(r for r in iterator if r["label"] >= 0)
    if not rows:
        raise RuntimeError("No existing raw inputs. Dev mode never downloads; keep the existing dev_sample or create it on the teammate's machine.")
    rng = np.random.default_rng(config["seed"])
    selected = []
    target = config["data"]["sample_images"]
    # Balance classes, then sources within each class; impossible cells (NIH COVID)
    # are omitted. Select one image per known patient within each stratum.
    for label in range(3):
        class_rows = [r for r in rows if r["label"] == label]
        sources = sorted({r["dataset"] for r in class_rows})
        if not sources:
            raise RuntimeError(f"Cannot stratify dev sample: {CLASSES[label]} unavailable")
        for index, source in enumerate(sources):
            candidates = [r for r in class_rows if r["dataset"] == source]
            rng.shuffle(candidates)
            unique, seen = [], set()
            for row in candidates:
                patient = row["patient_id"] or row["image_id"]
                if patient not in seen:
                    unique.append(row)
                    seen.add(patient)
            quota = target // 3 // len(sources) + (index < (target // 3) % len(sources))
            if len(unique) < quota:
                raise RuntimeError(f"Need {quota} unique {source}/{CLASSES[label]} inputs; found {len(unique)}")
            selected.extend(unique[:quota])
    preprocess(config, records_override=selected)
    split(config)
    partition(config)
    size = sum(p.stat().st_size for p in base.rglob("*") if p.is_file())
    if size > 150 * 1024**2:
        raise RuntimeError(f"Dev sample exceeds 150 MiB: {size/1024**2:.1f} MiB")
    summary = {"profile": "dev", "config_hash": config_hash(config), "images": len(selected),
               "bytes": size, "classes": dict(Counter(r["label_name"] for r in selected)),
               "source_classes": dict(Counter(r["dataset"] + "/" + r["label_name"] for r in selected)),
               "missing_sources": missing, "selection": "seeded class/source strata from existing inputs only; not an evaluation cohort"}
    write_json(completed, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def verify_chexpert(config):
    settings = config["datasets"]["chexpert"]
    if not settings["enabled"]:
        print("CheXpert-small pending registration/files: no approved train.csv found. "
              "Preparing public sources; place the approved small release under <data-root>/raw/chexpert and rerun the same command.", flush=True)
        return {"status": "pending_registration_or_files"}
    root = Path(config["paths"]["raw"]) / settings.get("directory", "chexpert")
    records = []
    for csv_name in settings.get("csvs", [settings["csv"]]):
        if not (root / csv_name).is_file():
            raise RuntimeError(f"Incomplete CheXpert-small: missing {root/csv_name}. Supply the complete approved small release.")
        records.extend(adapters.chexpert(root, csv_name))
    if len(records) != settings["expected_images"]:
        raise ValueError(f"CheXpert-small: expected {settings['expected_images']} train+valid images, found {len(records)}")
    inventory = root / "ruralmed-checksums.csv"
    prior = {}
    if inventory.exists():
        with inventory.open(encoding="utf-8", newline="") as stream:
            prior = {r["path"]: r["sha256"] for r in csv.DictReader(stream)}
    temporary = inventory.with_suffix(".csv.part")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "sha256"])
        writer.writeheader()
        paths = {Path(r["raw_path"]) for r in records} | {root / s for s in settings.get("csvs", [settings["csv"]])}
        for path in sorted(paths):
            relative = path.relative_to(root).as_posix()
            digest = sha256(path)
            if relative in prior and prior[relative] != digest:
                raise ValueError(f"CheXpert checksum changed: {relative}")
            writer.writerow({"path": relative, "sha256": digest})
    temporary.replace(inventory)
    return {"status": "verified", "images": len(records), "inventory_sha256": sha256(inventory),
            "checksum_origin": "local receipt; no publisher signature"}


def unpack_chexpert(archive, folder, delete_archives=False):
    """Only unpack an already approved local ZIP; never acquire CheXpert."""
    archive, folder = Path(archive).resolve(), Path(folder).resolve()
    if not archive.is_file():
        raise RuntimeError(f"Approved CheXpert archive not found: {archive}")
    folder.mkdir(parents=True, exist_ok=True)
    digest = sha256(archive)
    prior = folder / "acquisition.json"
    if prior.exists() and json.loads(prior.read_text())["sha256"] != digest:
        raise ValueError("CheXpert archive changed; use a fresh data root")
    if not verified_extraction(folder):
        count = extract_verified(archive, folder)
        write_json(prior, {"dataset": "chexpert-small", "sha256": digest,
                           "members_crc_verified": count,
                           "inventory_sha256": sha256(folder / "extracted-files.csv")})
    if delete_archives and archive.is_relative_to(folder):
        archive.unlink()


def export_outputs(config, receipts):
    processed = Path(config["paths"]["processed"])
    output = Path(config["paths"]["partitions"])
    for name in ["preprocessing.json", "split-summary.json"]:
        shutil.copyfile(processed / name, output / name)
    for path in (processed / "splits").glob("*.csv"):
        (output / "splits").mkdir(exist_ok=True)
        shutil.copyfile(path, output / "splits" / path.name)
    primary = pd.read_csv(processed / "primary_manifest.csv", keep_default_na=False)
    primary[["image_id", "processed_path", "dataset", "label", "split"]].to_csv(output / "primary_index.csv", index=False)
    summary = {"profile": "full", "cohort": config["data"]["cohort"], "config_hash": config_hash(config),
               "sources": receipts, "images": len(primary), "chexpert_pending": not config["datasets"]["chexpert"]["enabled"],
               "paths": "Image paths are relative to the processed root; configure RURALMED_DATA_ROOT on each machine."}
    write_json(output / "preparation.json", summary)
    report = output / "report.md"
    text = report.read_text(encoding="utf-8")
    text += ("\n## Source verification\n\n" + ("CheXpert pending registration/files; public cohort only.\n" if summary["chexpert_pending"] else "All three sources verified.\n")
             + "\nSee preparation.json for pinned release hashes/counts and split-summary.json for leakage audits.\n")
    report.write_text(text, encoding="utf-8")
    return summary


def prepare_full(config, config_path, delete_archives=False, no_download=False, tracking=True, chexpert_archive=None):
    # Only allocations reused by this command count towards the budget. Legacy
    # pilot directories or segmented transfers cannot reduce the storage gate.
    if not no_download:
        raw = Path(config["paths"]["raw"])
        reused = [Path(config["paths"]["processed"]), raw / "chexpert"]
        for name, version in (("nih", 3), ("covidqu", 7)):
            reused.extend([raw / name / "extracted", raw / name / f"version-{version}.zip",
                           raw / name / f"version-{version}.zip.part"])
        existing = sum(path.stat().st_size if path.is_file() else
                       sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                       for path in reused)
        budget = max(config["data"]["minimum_free_gib"], 100 if delete_archives else 125)
        remaining = max(5, budget - existing / 1024**3)
        disk_preflight(Path(config["paths"]["raw"]).parent, remaining)
    chex_folder = Path(config["paths"]["raw"]) / "chexpert"
    local_archive = Path(chexpert_archive) if chexpert_archive else chex_folder / "CheXpert-v1.0-small.zip"
    if chexpert_archive or local_archive.is_file():
        unpack_chexpert(local_archive, chex_folder, delete_archives)
        config = load_config(config_path, profile="full")
    receipts = {"chexpert": verify_chexpert(config)}
    for source in ("nih", "covidqu"):
        receipts[source] = acquire(source, config["paths"]["raw"], delete_archives=delete_archives, allow_download=not no_download)
    if tracking:
        from .pipeline import run
        run(config_path, config=config)
    else:
        preprocess(config)
        split(config)
        partition(config)
    return export_outputs(config, receipts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["dev", "full"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-root", type=Path, help="Existing raw root for dev sampling only")
    parser.add_argument("--chexpert-archive", type=Path, help="Approved local CheXpert-small ZIP; never downloaded by this CLI")
    parser.add_argument("--delete-archives", action="store_true", help="Delete ZIPs inside the managed data root after verified extraction")
    parser.add_argument("--no-download", action="store_true", help="Use only verified existing extractions/archives")
    parser.add_argument("--no-tracking", action="store_true", help="Skip MLflow connection (leaves that Phase 1 gate pending)")
    args = parser.parse_args()
    profile = args.profile or "full"
    config_path = args.config or REPO_ROOT / f"research/configs/{'phase1-dev' if profile == 'dev' else 'phase1'}.yml"
    config = load_config(config_path, profile=args.profile)
    try:
        if config["data"]["profile"] == "dev":
            dev_sample(config, args.source_root)
        else:
            if args.source_root:
                parser.error("--source-root is for dev sampling only; set RURALMED_DATA_ROOT for full data")
            result = prepare_full(config, config_path, args.delete_archives, args.no_download, not args.no_tracking, args.chexpert_archive)
            destination = Path(config["paths"]["partitions"]).parent / result["cohort"]
            print(f"Preparation complete ({result['cohort']}); outputs: {destination}")
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Data preparation stopped: {exc}\nRerun the same command to resume after resolving the cause.\n")
