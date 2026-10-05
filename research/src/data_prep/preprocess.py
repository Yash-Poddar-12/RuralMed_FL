"""Verified raw inputs -> lossless PNGs + provenance manifest; entirely CPU."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from . import adapters
from .common import load_config, config_hash, write_json
from .acquire import sha256


def process_image(record, output, size):
    target = Path(output) / "images" / record["dataset"] / (record["image_id"].split(":")[1] + ".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = target.with_suffix(".verified.json")
    raw = Path(record["raw_path"])
    signature = {"bytes": raw.stat().st_size, "mtime_ns": raw.stat().st_mtime_ns}
    if receipt.exists() and target.exists():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if (saved["raw_stat"] == signature and saved["size"] == size
                and saved["source"] == record and sha256(target) == saved["file_sha256"]):
            return saved["record"]
    with Image.open(record["raw_path"]) as image:
        image.load()  # Decode completely; fail on corrupt inputs.
        gray = ImageOps.grayscale(image)
        raw_hash = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
        gray = gray.resize((size, size), Image.Resampling.BILINEAR)
        pixel_hash = hashlib.sha256(gray.tobytes()).hexdigest()
        verified = False
        if target.exists():
            try:
                with Image.open(target) as saved:
                    verified = saved.size == (size, size) and hashlib.sha256(saved.tobytes()).hexdigest() == pixel_hash
            except OSError:
                pass
        if not verified:
            temporary = target.with_name(target.name + ".part")
            gray.save(temporary, format="PNG", compress_level=1)
            temporary.replace(target)
    result = {**record, "processed_path": target.relative_to(output).as_posix(),
            "raw_pixel_sha256": raw_hash, "pixel_sha256": pixel_hash,
            "width": size, "height": size}
    write_json(receipt, {"raw_stat": signature, "size": size, "source": record,
                         "file_sha256": sha256(target), "record": result})
    return result


def run(config, records_override=None):
    root = Path(config["paths"]["raw"])
    output = Path(config["paths"]["processed"])
    output.mkdir(parents=True, exist_ok=True)
    previous = output / "preprocessing.json"
    if previous.exists() and json.loads(previous.read_text())["config_hash"] != config_hash(config):
        raise ValueError("Configuration changed; choose a fresh processed output directory")
    records, inventories = [], {}
    for name, settings in config["datasets"].items() if records_override is None else []:
        if not settings["enabled"]:
            continue
        dataset_root = root / settings.get("directory", name)
        if name in {"nih", "covidqu"}:
            receipt = dataset_root / "acquisition.json"
            if settings.get("representative_only"):
                receipt = dataset_root / "subset.json"
                subset = json.loads(receipt.read_text())
                if not subset.get("representative_only") or not subset.get("crc_verified"):
                    raise ValueError("Representative input lacks CRC verification")
            elif not receipt.exists():
                raise ValueError(f"{name}: acquisition verification receipt missing")
        iterator = (getattr(adapters, name)(dataset_root / "extracted") if name != "chexpert"
                    else (row for csv_name in settings.get("csvs", [settings["csv"]])
                          for row in adapters.chexpert(dataset_root, csv_name)))
        rows = list(iterator)
        if settings.get("expected_images") and len(rows) != settings["expected_images"]:
            raise ValueError(f"{name}: expected {settings['expected_images']} images, found {len(rows)}")
        counts = dict(Counter(r["label_name"] for r in rows))
        if settings.get("expected_classes") and counts != settings["expected_classes"]:
            raise ValueError(f"{name}: class inventory mismatch {counts}")
        inventories[name] = {"images": len(rows), "classes": counts}
        records.extend(rows)
    if records_override is not None:
        records = records_override
        for name in sorted({r["dataset"] for r in records}):
            rows = [r for r in records if r["dataset"] == name]
            inventories[name] = {"images": len(rows), "classes": dict(Counter(r["label_name"] for r in rows))}
    if not records:
        raise ValueError("No enabled data")
    processed = []
    with ThreadPoolExecutor(max_workers=config["preprocessing"]["workers"]) as pool:
        for i, row in enumerate(pool.map(lambda r: process_image(r, output, config["preprocessing"]["image_size"]), records), 1):
            processed.append(row)
            if i % 5000 == 0:
                print(f"Decoded and processed {i}/{len(records)}", flush=True)
    manifest = pd.DataFrame(processed).sort_values("image_id")
    manifest.to_csv(output / "manifest.csv", index=False)
    summary = {"config_hash": config_hash(config), "inventories": inventories,
               "profile": config.get("data", {}).get("profile", "full"),
               "representative_only": any(s.get("representative_only", False) for s in config["datasets"].values()),
               "images_decoded": len(manifest), "primary_images": int((manifest.label >= 0).sum()),
               "excluded_images": int((manifest.label < 0).sum()),
               "duplicate_processed_pixels": int(manifest.pixel_sha256.duplicated().sum()),
               "missing_patient_ids": int(manifest.patient_id.eq("").sum()),
               "normalization": "ImageNet at load time; grayscale replicated to RGB"}
    write_json(output / "preprocessing.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/configs/phase1.yml")
    run(load_config(parser.parse_args().config))
