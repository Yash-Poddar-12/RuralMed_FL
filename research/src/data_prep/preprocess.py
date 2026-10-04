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


def process_image(record, output, size):
    target = Path(output) / "images" / record["dataset"] / (record["image_id"].split(":")[1] + ".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(record["raw_path"]) as image:
        image.load()  # Decode completely; fail on corrupt inputs.
        gray = ImageOps.grayscale(image)
        raw_hash = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
        gray = gray.resize((size, size), Image.Resampling.BILINEAR)
        pixel_hash = hashlib.sha256(gray.tobytes()).hexdigest()
        if not target.exists():
            gray.save(target, compress_level=1)
        else:
            with Image.open(target) as saved:
                if saved.size != (size, size) or hashlib.sha256(saved.tobytes()).hexdigest() != pixel_hash:
                    raise ValueError(f"Processed image mismatch: {target}; use a fresh output directory")
    return {**record, "processed_path": target.relative_to(output).as_posix(),
            "raw_pixel_sha256": raw_hash, "pixel_sha256": pixel_hash,
            "width": size, "height": size}


def run(config):
    root = Path(config["paths"]["raw"])
    output = Path(config["paths"]["processed"])
    output.mkdir(parents=True, exist_ok=True)
    previous = output / "preprocessing.json"
    if previous.exists() and json.loads(previous.read_text())["config_hash"] != config_hash(config):
        raise ValueError("Configuration changed; choose a fresh processed output directory")
    records, inventories = [], {}
    for name, settings in config["datasets"].items():
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
        iterator = getattr(adapters, name)(dataset_root / "extracted") if name != "chexpert" else adapters.chexpert(dataset_root, settings["csv"])
        rows = list(iterator)
        if settings.get("expected_images") and len(rows) != settings["expected_images"]:
            raise ValueError(f"{name}: expected {settings['expected_images']} images, found {len(rows)}")
        counts = dict(Counter(r["label_name"] for r in rows))
        if settings.get("expected_classes") and counts != settings["expected_classes"]:
            raise ValueError(f"{name}: class inventory mismatch {counts}")
        inventories[name] = {"images": len(rows), "classes": counts}
        records.extend(rows)
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
