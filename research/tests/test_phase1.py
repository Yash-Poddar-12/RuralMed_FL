"""Check data-integrity boundaries and reproducible end-to-end CPU behavior."""
import copy
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from research.src.data_prep.adapters import nih_label, chexpert_label
from research.src.data_prep.common import load_config
from research.src.data_prep.preprocess import run as preprocess
from research.src.data_prep.split import run as split
from research.src.partitioning.partition import run as partition
from research.src.data_prep.dataset import ImagingDataset


def test_harmonization():
    assert nih_label("No Finding")[0] == 0
    for finding in ["Pneumonia", "Infiltration|Effusion", "Consolidation"]:
        assert nih_label(finding)[0] == 1
    assert nih_label("Cardiomegaly")[0] == -1
    assert chexpert_label({"Pneumonia": "-1", "No Finding": "1"})[0] == -1
    assert chexpert_label({"Pneumonia": "-1", "Consolidation": "1"})[0] == 1
    assert chexpert_label({"Pneumonia": "", "Consolidation": "0", "No Finding": "1"})[0] == 0
    assert chexpert_label({"No Finding": "1", "Effusion": "1"})[0] == -1


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase1")
    config = load_config("research/configs/phase1.yml")
    config["paths"] = {"raw": str(root / "raw"), "processed": str(root / "processed"), "partitions": str(root / "partitions")}
    config["preprocessing"]["workers"] = 4
    config["datasets"]["nih"]["expected_images"] = 600
    config["datasets"]["covidqu"]["expected_images"] = 900
    config["datasets"]["covidqu"]["expected_classes"] = {"Normal": 300, "Pneumonia": 300, "COVID-19": 300}
    rng = np.random.default_rng(2026)
    rows = []
    nih_root = root / "raw" / "nih" / "extracted"
    nih_root.mkdir(parents=True)
    for i in range(600):
        name = f"{i:08d}.png"
        Image.fromarray(rng.integers(0, 256, (24, 24), dtype=np.uint8)).save(nih_root / name)
        rows.append({"Image Index": name, "Patient ID": str(i // 3), "Finding Labels": "No Finding" if i // 3 % 2 else "Infiltration"})
    with (nih_root / "Data_Entry_2017.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    covid_root = root / "raw" / "covidqu" / "extracted" / "Lung Segmentation Data" / "Lung Segmentation Data"
    for cls in ["Normal", "Non-COVID", "COVID-19"]:
        for i in range(300):
            split_name = "Train" if i < 210 else "Val" if i < 255 else "Test"
            folder = covid_root / split_name / cls / "images"
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (24, 24), dtype=np.uint8)).save(folder / f"{i}.png")
    # Acquisition receipts are fixture placeholders, never presented as real data.
    for name in ["nih", "covidqu"]:
        (root / "raw" / name / "acquisition.json").write_text('{"test_fixture":true}')
    preprocess(config)
    primary = split(config)
    result = partition(config)
    return config, primary, result


def test_group_isolation_and_coverage(prepared):
    config, data, report = prepared
    assert data.groupby("group_id").split.nunique().max() == 1
    known = data[data.patient_id != ""]
    assert known.groupby("patient_id").split.nunique().max() == 1
    assignments = pd.read_csv(Path(config["paths"]["partitions"]) / "assignments.csv")
    assert not assignments.image_id.duplicated().any()
    assert set(assignments.image_id) == set(data[data.split != "test"].image_id)
    assert not set(assignments.image_id) & set(data[data.split == "test"].image_id)
    assert assignments.groupby("group_id").client.nunique().max() == 1
    assert report["min_client_train"] >= 32
    assert report["volume_ratio"] > 2


def test_repeat_partition(prepared, tmp_path):
    config, _, _ = prepared
    second = copy.deepcopy(config)
    second["paths"]["partitions"] = str(tmp_path)
    partition(second)
    for path in Path(config["paths"]["partitions"]).glob("client_*/*.csv"):
        assert path.read_bytes() == (tmp_path / path.parent.name / path.name).read_bytes()


def test_cpu_loader_and_unknown_index(prepared, tmp_path):
    config, _, _ = prepared
    index = Path(config["paths"]["partitions"]) / "client_00" / "train.csv"
    dataset = ImagingDataset(config["paths"]["processed"], index)
    for i in range(len(dataset)):
        image, label = dataset[i]
        assert image.dtype == torch.float32 and image.device.type == "cpu"
        assert image.shape == (3, 224, 224) and torch.isfinite(image).all()
        assert label in [0, 1, 2]
    bad = tmp_path / "bad.csv"
    bad.write_text("image_id\nunknown\n")
    with pytest.raises(ValueError, match="unknown"):
        ImagingDataset(config["paths"]["processed"], bad)


def test_inventory_mismatch(prepared):
    config, _, _ = prepared
    bad = copy.deepcopy(config)
    bad["datasets"]["nih"]["expected_images"] += 1
    # Different config protection is also an integrity gate.
    with pytest.raises(ValueError):
        preprocess(bad)


def test_duplicate_anchor_through_excluded_image(tmp_path):
    config = load_config("research/configs/phase1.yml")
    config["paths"]["processed"] = str(tmp_path)
    rows = [
        {"image_id": "nih:excluded", "patient_id": "nih:1", "dataset": "nih", "label": -1, "label_name": "Excluded", "pixel_sha256": "shared", "official_split": ""},
        {"image_id": "nih:primary", "patient_id": "nih:1", "dataset": "nih", "label": 0, "label_name": "Normal", "pixel_sha256": "other", "official_split": ""},
        {"image_id": "covid:copy", "patient_id": "", "dataset": "covidqu", "label": 0, "label_name": "Normal", "pixel_sha256": "shared", "official_split": "test"},
        {"image_id": "covid:train", "patient_id": "", "dataset": "covidqu", "label": 1, "label_name": "Pneumonia", "pixel_sha256": "train", "official_split": "train"},
        {"image_id": "covid:val", "patient_id": "", "dataset": "covidqu", "label": 2, "label_name": "COVID-19", "pixel_sha256": "val", "official_split": "val"},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    result = split(config).set_index("image_id")
    assert result.loc["nih:primary", "split"] == "test"
    assert result.loc["nih:primary", "group_id"] == result.loc["covid:copy", "group_id"]
