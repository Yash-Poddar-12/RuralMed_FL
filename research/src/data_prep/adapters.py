"""Each adapter returns the same metadata records, including excluded findings."""
import csv
import hashlib
import json
from pathlib import Path
from .common import CLASSES


def nih_label(findings):
    labels = set(findings.split("|"))
    if labels == {"No Finding"}:
        return 0, ""
    if labels & {"Pneumonia", "Infiltration", "Consolidation"}:
        return 1, ""
    return -1, "outside_primary_taxonomy"


def chexpert_label(row):
    values = {key: float(value) for key, value in row.items()
              if value not in (None, "") and key not in
              {"Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"}}
    targets = [values.get(key) for key in ["Pneumonia", "Consolidation"]]
    if any(value == 1 for value in targets):
        return 1, ""
    if values.get("No Finding") == 1 and not any(value == -1 for value in targets):
        if not any(value == 1 for key, value in values.items() if key != "No Finding"):
            return 0, ""
    return -1, "uncertain_target" if -1 in targets else "outside_primary_taxonomy"


def record(source, path, relative, patient, original, label, reason="", official=""):
    return dict(image_id=source + ":" + hashlib.sha256(relative.encode()).hexdigest()[:24],
                dataset=source, raw_path=str(path), raw_relative=relative,
                patient_id=patient, original_labels=json.dumps(original, sort_keys=True),
                label=label, label_name=CLASSES[label] if label >= 0 else "Excluded",
                exclusion_reason=reason, official_split=official)


def nih(root):
    root = Path(root)
    csvs = list(root.rglob("Data_Entry*.csv"))
    if len(csvs) != 1:
        raise ValueError(f"Expected one NIH metadata CSV, found {len(csvs)}")
    images = {}
    for path in root.rglob("*.png"):
        if path.name in images:
            raise ValueError(f"Duplicate NIH filename: {path.name}")
        images[path.name] = path
    with csvs[0].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [r["Image Index"] for r in rows]
    if len(names) != len(set(names)) or set(names) != set(images):
        raise ValueError("NIH metadata/image inventory mismatch")
    for row in rows:
        path = images[row["Image Index"]]
        label, reason = nih_label(row["Finding Labels"])
        yield record("nih", path, path.relative_to(root).as_posix(),
                     "nih:" + row["Patient ID"], row, label, reason)


def covidqu(root):
    root = Path(root)
    matches = [path for path in root.rglob("Lung Segmentation Data")
               if any(child.is_dir() and child.name.lower() in {"train", "val", "validation", "test"}
                      for child in path.iterdir())]
    if len(matches) != 1:
        raise ValueError(f"Expected full Lung Segmentation Data, found {len(matches)}")
    mapping = {"normal": 0, "non-covid": 1, "non-covid-19": 1, "covid-19": 2, "covid": 2}
    splits = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
    for path in sorted(matches[0].rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        parts = [p.lower() for p in path.relative_to(matches[0]).parts[:-1]]
        if "images" not in parts:
            continue  # Never consume lung masks or the overlapping infection subset.
        found = [mapping[p] for p in parts if p in mapping]
        official = [splits[p] for p in parts if p in splits]
        if len(found) != 1 or len(official) != 1:
            raise ValueError(f"Unrecognized COVID-QU-Ex layout: {path}")
        yield record("covidqu", path, path.relative_to(root).as_posix(), "",
                     {"directory_class": CLASSES[found[0]]}, found[0], official=official[0])


def chexpert(root, csv_name="train.csv"):
    root = Path(root)
    with (root / csv_name).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            relative = Path(row["Path"])
            candidates = [root / relative, root.parent / relative, root / Path(*relative.parts[1:])]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(row["Path"])
            patient = next((part for part in relative.parts if part.startswith("patient")), None)
            if patient is None:
                raise ValueError("CheXpert path lacks patient ID")
            label, reason = chexpert_label(row)
            if row.get("Frontal/Lateral", "Frontal") != "Frontal":
                label, reason = -1, "lateral_view"
            yield record("chexpert", path, relative.as_posix(), "chexpert:" + patient, row, label, reason)
