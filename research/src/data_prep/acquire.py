"""Public, version-pinned Kaggle acquisition; no credentials required."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

SOURCES = {"nih": "nih-chest-xrays/data", "covidqu": "anasmohammedtahir/covidqu"}


def segmented_download(url, archive, workers=24):
    """Resumable, bounded parallel ranges; final ZIP remains byte-identical."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError("Source does not support ranged transfer")
        total = int(response.headers["Content-Range"].split("/")[-1])
        direct = response.url
        response.read()
    if shutil.disk_usage(archive.parent).free < total * 2 + 5 * 1024**3:
        raise RuntimeError("Insufficient space for segments plus assembled archive")
    chunks = archive.with_suffix(".segments")
    chunks.mkdir(exist_ok=True)
    size = 64 * 1024 * 1024
    # Reuse complete ranges from an earlier sequential transfer on restart.
    sequential = archive.with_suffix(".zip.part")
    if sequential.exists():
        with sequential.open("rb") as stream:
            for i in range(sequential.stat().st_size // size):
                block = stream.read(size)
                target = chunks / f"{i:05d}"
                if not target.exists():
                    target.write_bytes(block)
    tasks = [(i, start, min(start + size, total) - 1) for i, start in enumerate(range(0, total, size))]
    done = 0
    def fetch(task):
        i, start, end = task
        target = chunks / f"{i:05d}"
        expected = end - start + 1
        if target.exists() and target.stat().st_size == expected:
            return expected
        part = target.with_suffix(".part")
        if part.exists() and part.stat().st_size == expected:
            part.replace(target)
            return expected
        source_url = direct
        for attempt in range(6):
            offset = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(source_url, headers={"Range": f"bytes={start + offset}-{end}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start + offset}-{end}/{total}":
                        raise RuntimeError("Range response mismatch")
                    with part.open("ab") as out:
                        shutil.copyfileobj(response, out, 1024 * 1024)
                if part.stat().st_size != expected:
                    raise RuntimeError("Incomplete range")
                part.replace(target)
                return expected
            except (OSError, RuntimeError):
                source_url = url  # Refresh signed redirect on retry/expiry.
                if attempt == 5:
                    raise
                time.sleep(2)
    from concurrent.futures import as_completed
    last = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, task) for task in tasks]):
            done += future.result()
            if time.monotonic() - last > 30:
                print(f"{archive.parent.name}: segments verified-size {done / 1024**3:.2f}/{total / 1024**3:.2f} GiB", flush=True)
                last = time.monotonic()
    assembled = archive.with_suffix(".assembling")
    with assembled.open("wb") as output:
        for i, _, _ in tasks:
            with (chunks / f"{i:05d}").open("rb") as stream:
                shutil.copyfileobj(stream, output, 8 * 1024 * 1024)
    assert assembled.stat().st_size == total
    assembled.replace(archive)
    # Explicitly checked workspace-local transient segment directory only.
    if not chunks.resolve().is_relative_to(archive.parent.resolve()):
        raise ValueError("Segment directory escaped raw dataset root")
    shutil.rmtree(chunks)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acquire(name, root, segmented=False):
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    ref = SOURCES[name]
    api = "https://www.kaggle.com/api/v1/datasets/"
    metadata_path = folder / "source.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(api + "view/" + ref, timeout=60) as response:
            metadata = json.load(response)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    version = metadata["currentVersionNumber"]
    archive = folder / f"version-{version}.zip"
    receipt = folder / "acquisition.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if archive.exists() and sha256(archive) == saved["sha256"]:
            print(f"{name}: already acquired and archive hash verified", flush=True)
            return saved
        raise ValueError(f"{name}: existing receipt disagrees with archive")
    url = api + f"download/{ref}?datasetVersionNumber={version}"
    partial = archive.with_suffix(".zip.part")
    if segmented and not archive.exists():
        segmented_download(url, archive)
    if not archive.exists():
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if "text/html" in response.headers.get("Content-Type", ""):
                        raise RuntimeError(f"{name}: server returned HTML instead of archive")
                    resumed = offset > 0 and response.status == 206
                    length = int(response.headers.get("Content-Length", 0))
                    if shutil.disk_usage(folder).free < length + 8 * 1024**3:
                        raise RuntimeError(f"{name}: insufficient free disk for download plus reserve")
                    total = offset if resumed else 0
                    last = time.monotonic()
                    with partial.open("ab" if resumed else "wb") as stream:
                        while block := response.read(8 * 1024 * 1024):
                            stream.write(block)
                            total += len(block)
                            if time.monotonic() - last > 30:
                                print(f"{name}: downloaded {total / 1024**3:.2f} GiB", flush=True)
                                last = time.monotonic()
                    if length and total - (offset if resumed else 0) != length:
                        raise RuntimeError("Incomplete response")
                partial.replace(archive)
                break
            except (OSError, RuntimeError) as exc:
                if attempt == 4:
                    raise
                print(f"{name}: retry {attempt + 1}: {exc}", flush=True)
                time.sleep(2)
    print(f"{name}: verifying and extracting archive", flush=True)
    extracted = folder / "extracted"
    extracted.mkdir(exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as z:
        needed = sum(m.file_size for m in z.infolist() if not (extracted / m.filename).exists())
        if shutil.disk_usage(folder).free < needed + 5 * 1024**3:
            raise RuntimeError(f"{name}: need {needed / 1024**3:.1f} GiB to extract plus 5 GiB reserve")
        for member in z.infolist():
            destination = (extracted / member.filename).resolve()
            if not destination.is_relative_to(extracted.resolve()):
                raise ValueError("Unsafe archive path")
            if member.is_dir():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Read every member, including on restart; ZipExtFile checks its CRC.
            with z.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            count += 1
            if count % 10000 == 0:
                print(f"{name}: verified {count} archive members", flush=True)
    result = {"dataset": name, "ref": ref, "version": version,
              "download_url": url, "downloaded_at": datetime.now(timezone.utc).isoformat(),
              "sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
              "members_crc_verified": count, "source_updated": metadata.get("lastUpdated")}
    receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{name}: acquisition complete: {result}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="research/data/raw")
    parser.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--segmented", action="store_true")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=len(args.datasets)) as pool:
        futures = [pool.submit(acquire, name, args.root, args.segmented) for name in args.datasets]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
