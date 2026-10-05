"""Pinned public releases, restartable downloads and verified extraction.

Checksums are local integrity receipts, not publisher signatures. No credentials
or CheXpert downloads are attempted here. Use prepare_data.py for orchestration.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import urllib.error
import zipfile
import zlib

from .common import write_json

SOURCES = {"nih": ("nih-chest-xrays/data", 3),
           "covidqu": ("anasmohammedtahir/covidqu", 7)}
COVID_SHA256 = "2a91b372cdd104d05d472f79dc446dc9bb7e3ccc26f513cf971f64cabc151333"
GIB = 1024 ** 3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def disk_preflight(root, minimum_gib=100):
    probe = Path(root).resolve()
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free / GIB
    if free < minimum_gib:
        raise RuntimeError(f"Data disk has {free:.1f} GiB free; need at least {minimum_gib:.1f} GiB "
                           f"before downloading. Set RURALMED_DATA_ROOT to a larger drive ({root}).")
    print(f"Disk preflight: {free:.1f} GiB free at {probe}", flush=True)


def download(url, archive):
    partial = archive.with_suffix(".zip.part")
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise RuntimeError("Source returned HTML, not an archive")
                resumed = offset > 0 and response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Download range mismatch")
                length = int(response.headers.get("Content-Length", 0))
                if shutil.disk_usage(archive.parent).free < length + 5 * GIB:
                    raise RuntimeError("Insufficient space for remaining archive plus 5 GiB reserve")
                received = 0
                last = time.monotonic()
                with partial.open("ab" if resumed else "wb") as stream:
                    while block := response.read(8 * 1024 * 1024):
                        stream.write(block)
                        received += len(block)
                        if time.monotonic() - last > 30:
                            print(f"{archive.parent.name}: downloaded {(received + (offset if resumed else 0))/GIB:.2f} GiB", flush=True)
                            last = time.monotonic()
                if length and received != length:
                    raise RuntimeError("Incomplete download; rerun to resume")
            partial.replace(archive)
            return
        except (OSError, RuntimeError) as exc:
            if (isinstance(exc, urllib.error.HTTPError) and exc.code == 416
                    and exc.headers.get("Content-Range") == f"bytes */{offset}" and offset):
                partial.replace(archive)
                return
            if attempt == 5:
                raise
            time.sleep(2)


def verified_extraction(folder):
    """Check every extracted file against its durable checksum inventory."""
    receipt = folder / "acquisition.json"
    inventory = folder / "extracted-files.csv"
    if not receipt.exists() or not inventory.exists():
        return None
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    if sha256(inventory) != saved.get("inventory_sha256"):
        return None
    count = 0
    with inventory.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            path = (folder / "extracted" / row["path"]).resolve()
            if not path.is_relative_to((folder / "extracted").resolve()):
                raise ValueError("Unsafe extraction inventory path")
            if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
                return None
            count += 1
    return saved if count == saved["members_crc_verified"] else None


def file_crc_sha(path):
    crc = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
            digest.update(block)
    return crc, digest.hexdigest()


def extract_verified(archive, folder):
    extracted = folder / "extracted"
    extracted.mkdir(exist_ok=True)
    temporary = folder / "extracted-files.csv.part"
    count = 0
    with zipfile.ZipFile(archive) as source, temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        needed = sum(m.file_size for m in source.infolist()
                     if not m.is_dir() and not (extracted / m.filename).is_file())
        if shutil.disk_usage(folder).free < needed + 5 * GIB:
            raise RuntimeError(f"Need {needed/GIB:.1f} GiB for extraction plus 5 GiB reserve")
        for member in source.infolist():
            target = (extracted / member.filename).resolve()
            if not target.is_relative_to(extracted.resolve()):
                raise ValueError("Unsafe archive path")
            if member.is_dir():
                continue
            digest = None
            if target.is_file() and target.stat().st_size == member.file_size:
                crc, candidate = file_crc_sha(target)
                if crc == member.CRC:
                    digest = candidate
            if digest is None:
                target.parent.mkdir(parents=True, exist_ok=True)
                part = target.with_name(target.name + ".extracting")
                with source.open(member) as src, part.open("wb") as out:
                    shutil.copyfileobj(src, out, 1024 * 1024)  # ZipExtFile checks CRC.
                part.replace(target)
                digest = sha256(target)
            writer.writerow({"path": member.filename, "bytes": member.file_size, "sha256": digest})
            count += 1
            if count % 10000 == 0:
                print(f"{folder.name}: verified {count} archive members", flush=True)
    temporary.replace(folder / "extracted-files.csv")
    return count


def acquire(name, root, delete_archives=False, allow_download=True):
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    ref, version = SOURCES[name]
    archive = folder / f"version-{version}.zip"
    saved = verified_extraction(folder)
    if saved and (saved.get("ref"), saved.get("version")) == (ref, version):
        if archive.exists():
            if sha256(archive) != saved["sha256"]:
                raise ValueError(f"{name}: archive checksum mismatch")
            if delete_archives:
                archive.unlink()
        print(f"{name}: verified extracted checksums; skipping acquisition", flush=True)
        return saved
    if not archive.exists():
        if not allow_download:
            raise RuntimeError(f"{name}: verified extraction or archive required (--no-download)")
        # No HTTP request may occur before the storage gate.
        disk_preflight(folder, 45 if name == "nih" else 3)
        url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}?datasetVersionNumber={version}"
        download(url, archive)
    digest = sha256(archive)
    receipt = folder / "acquisition.json"
    if receipt.exists():
        prior = json.loads(receipt.read_text(encoding="utf-8"))
        if prior.get("version") == version and prior.get("sha256") != digest:
            raise ValueError(f"{name}: archive differs from saved SHA-256 receipt")
    if name == "covidqu" and digest != COVID_SHA256:
        raise ValueError("COVID-QU-Ex v7 SHA-256 differs from the recorded verified release")
    count = extract_verified(archive, folder)
    result = {"dataset": name, "ref": ref, "version": version, "sha256": digest,
              "archive_bytes": archive.stat().st_size, "members_crc_verified": count,
              "inventory_sha256": sha256(folder / "extracted-files.csv"),
              "verified_at": datetime.now(timezone.utc).isoformat()}
    write_json(receipt, result)
    if delete_archives:
        archive.unlink()
    return result


if __name__ == "__main__":
    raise SystemExit("Use python research/scripts/prepare_data.py --profile full; see infra/docs/phase1-handoff.md")
