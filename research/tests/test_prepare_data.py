"""Offline regression gates: no test may download real datasets."""
import json
import io
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pandas as pd
import pytest
import torch

from research.src.data_prep import acquire as acquisition
from research.src.data_prep import prepare
from research.src.data_prep.common import REPO_ROOT, config_hash, load_config
from research.src.data_prep.dataset import ImagingDataset
from research.src.data_prep.preprocess import process_image


def test_preflight_before_network(tmp_path, monkeypatch):
    config = load_config(REPO_ROOT / "research/configs/phase1.yml")
    config["paths"]["raw"] = str(tmp_path / "raw")
    config["paths"]["processed"] = str(tmp_path / "processed")
    monkeypatch.setattr(acquisition.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024**3))
    def forbidden(*args, **kwargs):
        pytest.fail("Network was called before disk preflight")
    monkeypatch.setattr(acquisition.urllib.request, "urlopen", forbidden)
    with pytest.raises(RuntimeError, match="before downloading"):
        prepare.prepare_full(config, "unused.yml")


def test_extract_resume_delete_and_corruption(tmp_path, monkeypatch):
    folder = tmp_path / "nih"
    folder.mkdir()
    archive = folder / "version-3.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("images/example.txt", "verified fixture")
    monkeypatch.setattr(acquisition.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3))
    def forbidden(*args, **kwargs):
        pytest.fail("Fixture acquisition attempted a download")
    monkeypatch.setattr(acquisition.urllib.request, "urlopen", forbidden)
    saved = acquisition.acquire("nih", tmp_path, allow_download=False)
    image = folder / "extracted/images/example.txt"
    timestamp = image.stat().st_mtime_ns
    assert acquisition.acquire("nih", tmp_path, allow_download=False) == saved
    assert image.stat().st_mtime_ns == timestamp
    image.write_text("corrupted data")
    acquisition.acquire("nih", tmp_path, allow_download=False, delete_archives=True)
    assert image.read_text() == "verified fixture"
    assert not archive.exists()
    assert acquisition.acquire("nih", tmp_path, allow_download=False)["sha256"] == saved["sha256"]
    image.write_text("broken again")
    with pytest.raises(RuntimeError, match="--no-download"):
        acquisition.acquire("nih", tmp_path, allow_download=False)


def test_archive_escape_rejected(tmp_path, monkeypatch):
    folder = tmp_path / "nih"
    folder.mkdir()
    archive = folder / "version-3.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../outside.txt", "unsafe")
    monkeypatch.setattr(acquisition.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3))
    with pytest.raises(ValueError, match="Unsafe archive"):
        acquisition.acquire("nih", tmp_path, allow_download=False)
    assert not (tmp_path / "outside.txt").exists()


def test_data_root_and_hash_are_portable(tmp_path, monkeypatch):
    monkeypatch.setenv("RURALMED_DATA_ROOT", str(tmp_path / "drive-a"))
    first = load_config(REPO_ROOT / "research/configs/phase1.yml")
    monkeypatch.setenv("RURALMED_DATA_ROOT", str(tmp_path / "drive-b"))
    second = load_config(REPO_ROOT / "research/configs/phase1.yml")
    assert first["paths"]["raw"] == str(tmp_path / "drive-a/raw")
    assert second["paths"]["processed"] == str(tmp_path / "drive-b/processed/full/public")
    assert config_hash(first) == config_hash(second)
    dev = load_config(REPO_ROOT / "research/configs/phase1-dev.yml")
    assert "dev_sample" in dev["paths"]["processed"]


def test_missing_chexpert_is_explicit(capsys):
    config = load_config(REPO_ROOT / "research/configs/phase1-dev.yml")
    assert prepare.verify_chexpert(config)["status"] == "pending_registration_or_files"
    assert "pending registration/files" in capsys.readouterr().out


def test_local_dev_sample_and_rerun(monkeypatch):
    config = load_config(REPO_ROOT / "research/configs/phase1-dev.yml")
    root = Path(config["paths"]["processed"])
    if not (root.parent / "sample.json").exists():
        pytest.skip("Create the offline dev sample from existing raw inputs first")
    manifest = prepare.verify_processed(config)
    assert len(manifest) == 300
    assert set(manifest.label) == {0, 1, 2}
    assert set(manifest.dataset) >= {"nih", "covidqu"}
    assert sum(p.stat().st_size for p in root.parent.rglob("*") if p.is_file()) < 150 * 1024**2
    for client in range(10):
        index = Path(config["paths"]["partitions"]) / f"client_{client:02d}/train.csv"
        data = ImagingDataset(root, index)
        for image, label in data:
            assert image.shape == (3, 224, 224) and image.dtype == torch.float32
            assert torch.isfinite(image).all() and label in range(3)
    def forbidden(*args, **kwargs):
        pytest.fail("Dev mode called network")
    monkeypatch.setattr(acquisition.urllib.request, "urlopen", forbidden)
    before = {p: p.read_bytes() for p in Path(config["paths"]["partitions"]).glob("client_*/*.csv")}
    prepare.dev_sample(config, source_root=Path("a-source-that-does-not-exist"))
    assert all(p.read_bytes() == content for p, content in before.items())


def test_processed_image_cache_skips_decode(tmp_path, monkeypatch):
    from PIL import Image
    import numpy as np
    raw = tmp_path / "raw.png"
    Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(raw)
    row = {"dataset": "nih", "image_id": "nih:fixture", "raw_path": str(raw)}
    first = process_image(row, tmp_path / "processed", 224)
    (tmp_path / "processed" / first["processed_path"]).write_bytes(b"interrupted output")
    assert process_image(row, tmp_path / "processed", 224) == first
    def forbidden(*args, **kwargs):
        pytest.fail("Verified output was decoded again")
    monkeypatch.setattr(Image, "open", forbidden)
    assert process_image(row, tmp_path / "processed", 224) == first


def test_download_resumes_exact_range(tmp_path, monkeypatch):
    archive = tmp_path / "release.zip"
    archive.with_suffix(".zip.part").write_bytes(b"abc")
    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 3-5/6", "Content-Length": "3"}
    def respond(request, **kwargs):
        assert request.get_header("Range") == "bytes=3-"
        return Response(b"def")
    monkeypatch.setattr(acquisition.urllib.request, "urlopen", respond)
    monkeypatch.setattr(acquisition.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3))
    acquisition.download("https://fixture.invalid/release", archive)
    assert archive.read_bytes() == b"abcdef"


def test_complete_partial_download_is_promoted(tmp_path, monkeypatch):
    archive = tmp_path / "release.zip"
    archive.with_suffix(".zip.part").write_bytes(b"complete")
    def completed(request, **kwargs):
        raise acquisition.urllib.error.HTTPError(request.full_url, 416, "complete", {"Content-Range": "bytes */8"}, None)
    monkeypatch.setattr(acquisition.urllib.request, "urlopen", completed)
    acquisition.download("https://fixture.invalid/release", archive)
    assert archive.read_bytes() == b"complete"


def test_portable_export(tmp_path):
    config = load_config(REPO_ROOT / "research/configs/phase1-dev.yml")
    processed = tmp_path / "processed"
    output = tmp_path / "partitions"
    processed.mkdir()
    output.mkdir()
    (processed / "splits").mkdir()
    config["paths"].update(processed=str(processed), partitions=str(output))
    config["data"]["cohort"] = "public"
    for name in ["preprocessing.json", "split-summary.json"]:
        (processed / name).write_text("{}")
    for name in ["train", "val", "test"]:
        (processed / "splits" / f"{name}.csv").write_text("image_id\nnih:1\n")
    pd.DataFrame([{"image_id": "nih:1", "processed_path": "images/nih/1.png", "dataset": "nih",
                   "label": 0, "split": "test", "raw_path": "private-drive/raw/image.png",
                   "patient_id": "private-patient"}]).to_csv(processed / "primary_manifest.csv", index=False)
    (output / "report.md").write_text("# Fixture report\n")
    result = prepare.export_outputs(config, {"nih": {"sha256": "fixture"}})
    assert result["chexpert_pending"]
    assert list(pd.read_csv(output / "primary_index.csv").columns) == ["image_id", "processed_path", "dataset", "label", "split"]
    assert "private-" not in (output / "primary_index.csv").read_text()
    assert (output / "splits/test.csv").exists()
    assert "CheXpert pending" in (output / "report.md").read_text()


def test_approved_chexpert_archive_is_local(tmp_path, monkeypatch):
    folder = tmp_path / "raw/chexpert"
    folder.mkdir(parents=True)
    archive = folder / "CheXpert-v1.0-small.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("CheXpert-v1.0-small/train.csv", "Path,No Finding\n")
    monkeypatch.setattr(acquisition.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3))
    def forbidden(*args, **kwargs):
        pytest.fail("CheXpert tried to download")
    monkeypatch.setattr(acquisition.urllib.request, "urlopen", forbidden)
    prepare.unpack_chexpert(archive, folder, delete_archives=True)
    assert not archive.exists()
    assert (folder / "extracted/CheXpert-v1.0-small/train.csv").exists()
    assert acquisition.verified_extraction(folder)["dataset"] == "chexpert-small"
