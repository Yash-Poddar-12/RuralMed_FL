# Research workspace

This directory contains the reproducible Python/ML workflow. Raw and processed medical images are local-only and ignored by Git; small configuration, metadata, and index artifacts are tracked where licensing permits.

For an interrupted setup, `python research/scripts/resume_phase1.py` resumes COVID acquisition, a representative-data CPU pilot, the pinned clean environment, and finally full NIH preparation. On this Windows machine it is running as a hidden background worker. Read `research/artifacts/continuation.json` for its current stage; `failed` includes the exact error. The worker leaves visual review pending even when all automated data gates pass. Its initial host audit uses the pre-existing Python 3.12/CUDA-enabled torch 2.5.1 on CPU; this is a diagnostic audit, while the separate Python 3.11 environment verification establishes the actual pinned environment.

For a detached Windows restart, run `powershell -NoProfile -File research/scripts/start_phase1.ps1` once. Check that no existing continuation worker is running before launching another. The helper's CUDA wheel bootstrap is specific to Windows CPython 3.11; use the normal install commands below on a teammate's Linux machine.

`research/configs/phase1-representative.yml` explicitly uses 1,500 CRC-verified NIH images from the downloaded ZIP prefix plus the complete COVID dataset. This ordered NIH subset is for pipeline validation, not a representative clinical evaluation cohort. Full release preparation uses `phase1.yml`.

## Install from scratch

Use Python 3.11 on Windows or Linux. From the repository root:

```powershell
conda env create -f research/environment.yml
conda activate fl-rural-imaging
python -m pip check
```

Pip fallback (same CUDA-enabled pins):

```powershell
py -3.11 -m venv research/.venv
research/.venv/Scripts/python -m pip install -r research/requirements.txt
research/.venv/Scripts/python -m pip check
```

On Linux, replace `Scripts/python` with `bin/python`. CUDA wheels run on CPU without a CUDA driver. GPU training later requires a compatible NVIDIA GPU/driver; Phase 1 performs no GPU operations. The official [PyTorch version table](https://pytorch.org/get-started/previous-versions/) documents the 2.7.1 / 0.22.1 CUDA 12.6 pairing. When installing with uv, add `--index-strategy unsafe-best-match` so pins are resolved across the trusted PyPI and PyTorch indexes. Keep caches on a drive with enough space (this machine uses D:).

## Public data and full CPU preparation

```powershell
python -m research.src.data_prep.acquire --segmented
python -m research.src.data_prep.pipeline --config research/configs/phase1.yml
python -m research.src.data_prep.inspect_client --client 0
python -m pytest research/tests -q
```

The download command pins source versions at first acquisition, supports restarts and verifies ZIP CRCs during extraction. Raw images are fully decoded and checked against published inventories before processing. Check free space first: NIH's archive and extracted copies together require about 84 GiB; leave room for CUDA wheels, COVID data and processed PNGs. No images or credentials enter Git. Re-running preprocessing validates existing outputs; change its output directory when changing config. Acquisition hashes, source versions, inventory counts, split audits and partition provenance are saved alongside local data.

The [data contract](../infra/docs/data-contract.md) describes harmonization, exact columns, duplicate policies and the Phase 2 loader. View `research/data/partitions/distribution.png` and `report.md` before training. The loader accepts any client's ID index:

```python
from research.src.data_prep.dataset import ImagingDataset
clinic = ImagingDataset("research/data/processed", "research/data/partitions/client_00/train.csv")
image, label = clinic[0]  # float32 CPU tensor [3,224,224], class ID 0/1/2
```

## Experiment tracking

The pipeline creates the local MLflow experiment `phase1-foundation`, logs the exact YAML and config hash, dataset/split statistics and partition report/plot. It runs without a W&B or cloud account. Tracking storage: `research/mlflow.db`; artifacts: `research/artifacts/mlflow/`. Start the viewer from the repo root:

```powershell
mlflow ui --backend-store-uri sqlite:///research/mlflow.db --host 127.0.0.1 --port 5000
```

CheXpert remains pending registration. Put the approved small release and `train.csv` in `research/data/raw/chexpert/`, enable it in `phase1.yml`, select a fresh processed/partition output location, and rerun the pipeline. Retain prior data/config hashes for comparability. No federated learning, compression, connectivity simulation or training logic is implemented in this phase.
