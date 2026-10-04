"""Durable Phase 1 continuation; write progress after each real verification gate."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from research.src.data_prep.acquire import acquire, segmented_download

STATUS = ROOT / "research/artifacts/continuation.json"
STATUS.parent.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "research/artifacts/continuation.log"
_log_stream = LOG.open("a", encoding="utf-8", buffering=1)
sys.stdout = _log_stream
sys.stderr = _log_stream

def status(stage, **extra):
    record = {"at": datetime.now(timezone.utc).isoformat(), "stage": stage, **extra}
    STATUS.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)
    if stage in {"verify-representative-on-host-cpu", "download-cuda-wheel", "tracked-representative-pipeline", "resume-full-nih-acquisition", "public-data-gates-passed-await-visual-review", "failed"}:
        message = extra.get("error", "").replace("\n", " ").replace("|", "/")[:500]
        with (ROOT / "infra/docs/decisions-log.md").open("a", encoding="utf-8") as log:
            log.write(f"\n| {record['at'][:10]} | Continuation gate | {stage}: {message} | Automatically recorded progress; gates that precede this stage have passed. Visual review remains manual. |\n")


def command(args):
    with LOG.open("a", encoding="utf-8") as log:
        subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    status("acquire-covidqu")
    acquire("covidqu", ROOT / "research/data/raw", segmented=True)
    status("verify-representative-on-host-cpu")
    command([sys.executable, "-m", "research.src.data_prep.preprocess", "--config", "research/configs/phase1-representative.yml"])
    command([sys.executable, "-m", "research.src.data_prep.split", "--config", "research/configs/phase1-representative.yml"])
    command([sys.executable, "-m", "research.src.partitioning.partition", "--config", "research/configs/phase1-representative.yml"])
    command([sys.executable, "-m", "research.src.data_prep.inspect_client", "--config", "research/configs/phase1-representative.yml", "--client", "0"])
    # Fetch the largest wheel with restartable ranges instead of discarding an
    # incomplete uv streaming download when an interactive turn is interrupted.
    status("download-cuda-wheel")
    wheel_dir = ROOT / ".cache/wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel = wheel_dir / "torch-2.7.1+cu126-cp311-cp311-win_amd64.whl"
    if not wheel.exists():
        segmented_download("https://download.pytorch.org/whl/cu126/torch-2.7.1%2Bcu126-cp311-cp311-win_amd64.whl", wheel, workers=16)
    status("install-pinned-environment")
    python = ROOT / "research/.venv/Scripts/python.exe"
    command(["uv", "--cache-dir", str(ROOT / ".cache/uv"), "pip", "install", "--python", str(python), "--index-strategy", "unsafe-best-match", str(wheel), "-r", "research/requirements.txt"])
    command(["uv", "--cache-dir", str(ROOT / ".cache/uv"), "pip", "check", "--python", str(python)])
    command([str(python), "research/scripts/verify_environment.py"])
    command([str(python), "-m", "pytest", "research/tests", "-q", "-o", "cache_dir=.cache/pytest-pinned", "--basetemp=.cache/test-tmp-pinned"])
    status("tracked-representative-pipeline")
    command([str(python), "-m", "research.src.data_prep.pipeline", "--config", "research/configs/phase1-representative.yml"])
    command([str(python), "-m", "research.src.data_prep.inspect_client", "--config", "research/configs/phase1-representative.yml", "--client", "0"])
    status("resume-full-nih-acquisition")
    acquire("nih", ROOT / "research/data/raw", segmented=True)
    status("full-public-data-pipeline")
    command([str(python), "-m", "research.src.data_prep.pipeline"])
    command([str(python), "-m", "research.src.data_prep.inspect_client", "--client", "0"])
    status("public-data-gates-passed-await-visual-review", chexpert="pending registration", supabase="pending team account", remote="pending destination")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        status("failed", error=traceback.format_exc())
        raise
