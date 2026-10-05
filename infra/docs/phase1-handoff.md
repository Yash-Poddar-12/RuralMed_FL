# Phase 1 teammate handoff

Snapshot: 2026-10-05. Read `Phase_Docs/PHASE 1.md` for the specification. Phase 1 is still pending full-data verification; no Phase 2 implementation has started. The continuation worker and both Windows launchers are retired and stop immediately if invoked. Full acquisition and preprocessing belong on the teammate's large drive. This laptop keeps an ignored 300-image dev sample.

## Clone and build

Share the latest local commits with `origin` before the teammate clones; this session commits locally and does not push.

```powershell
git clone https://github.com/Yash-Poddar-12/RuralMed_FL.git
Set-Location RuralMed_FL
conda env create -f research/environment.yml
conda activate fl-rural-imaging
python -m pip check
python research/scripts/verify_environment.py
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Python 3.11.9, torch 2.7.1+cu126 and torchvision 0.22.1+cu126 are pinned. `torch.cuda.is_available()` should be True on the GPU teammate's machine. Data preparation itself runs on CPU. False does not prevent Phase 1 preparation; resolve the GPU driver before later training. Existing Python 3.11 environment imports, CPU tensors and torchvision native operations were verified here; cold install time and a fresh teammate clone remain unmeasured. Allow roughly 15–45 minutes for environment installation depending on wheel/cache bandwidth.

Pip fallback on Windows (with Python 3.11 already installed):

```powershell
py -3.11 -m venv research/.venv
research/.venv/Scripts/python.exe -m pip install -r research/requirements.txt
research/.venv/Scripts/python.exe -m pip check
research/.venv/Scripts/python.exe research/scripts/verify_environment.py
```

Activate that venv before the commands below: `research/.venv/Scripts/Activate.ps1`. On Linux use `python3.11 -m venv research/.venv`, `source research/.venv/bin/activate`, and `python -m pip install -r research/requirements.txt`.

## Put data on the large drive

Start with about **100 GiB free** on a local SSD or large drive; the preflight requires **125 GiB** for a fresh run if retaining ZIPs (budget 125–150 GiB). Reserve additional space for the Python environment and package caches. NIH is approximately 42 GB compressed and needs a similarly sized extracted copy. The default download is sequential/resumable, avoiding a second assembled copy of segmented archives. `--delete-archives` removes managed ZIPs after CRC/checksum-verified extraction and substantially reduces peak/retained space.

```powershell
Copy-Item -LiteralPath .env.example -Destination .env
$env:RURALMED_DATA_ROOT = (Read-Host 'Full path to the large-drive data folder')
New-Item -ItemType Directory -Path $env:RURALMED_DATA_ROOT -Force
```

Put the same value in `RURALMED_DATA_ROOT=` in ignored `.env` to persist it, or export it each session. Exported values take precedence. Linux: `export RURALMED_DATA_ROOT=/your/large-drive/data`. No absolute machine path is stored in configs. Default if unset: `<repo>/research/data`. Full raw and processed files go beneath this root; committable indices/reports remain in the cloned repo.

CheXpert requires the team's Stanford registration/approval. The CLI never downloads it or bypasses access. Either place the approved **small** extracted release under `<data-root>/raw/chexpert/CheXpert-v1.0-small/` (containing train.csv, valid.csv, train/ and valid/), or place its approved ZIP at `<data-root>/raw/chexpert/CheXpert-v1.0-small.zip`. An already local ZIP elsewhere can be supplied with `--chexpert-archive <path>`; ZIPs outside the data root are retained. The CLI checks 223,414 training plus 234 validation images; official validation is held centrally as test, with conservative patient/duplicate grouping. It keeps all source records, including excluded/lateral findings, as metadata.

## Run once, unattended

```powershell
python research/scripts/prepare_data.py --profile full --delete-archives
```

Leave this foreground command running in the teammate's terminal. It requires no prompts and launches no background worker. It performs the disk gate before network access; downloads pinned NIH v3 and COVID-QU-Ex v7; verifies archive SHA-256, ZIP CRCs and durable extracted-file checksum inventories; checks published image/class counts; decodes and resizes every image; maps labels; groups patients/exact duplicates; creates splits; partitions train/validation across ten clinics; logs MLflow artifacts; and writes the report and portable indices. Checksums without a publisher signature are local reproducibility receipts; the recorded COVID v7 hash is also checked. NIH inventory must be 112,120; COVID inventory must be 33,920 with the documented class totals.

If CheXpert is absent, the command prints a clear pending-registration/files message and completes the `public` cohort. This is a successful public-data handoff, not a claim that all three datasets are complete. After approved CheXpert files arrive, rerun the identical command: it verifies/skips public acquisitions and builds a separate `all` cohort. An incomplete or corrupt present source fails clearly rather than silently dropping rows.

After interruption, rerun the same command. Existing allocations reduce the initial remaining-space budget; downloads append exact ranges, extraction skips CRC-verified files, and preprocessing skips outputs verified against their receipts. Interrupted PNG writes are atomic. ZIP deletion does not break restart because extracted inventories remain. `--no-download` enforces existing-files-only operation (it still permits extraction/processing). `--no-tracking` is available if MLflow is unavailable, but leaves the tracker gate pending.

Estimate **2–8 hours of CPU/SSD preparation**, plus transfer time: approximately **1–12 hours** for the large public download on a usable broadband connection. Slow connections or HDDs can require a day or longer. These are planning estimates, not measurements of a full run here. GPU speed does not accelerate the download. Keep the machine awake and connected.

## Verify, inspect and commit small outputs

The automatic `public` or `all` cohort determines the matching paths:

- `<data-root>/processed/full/<cohort>/`: PNGs, full/primary manifests, split indices and preprocessing receipts; never committed.
- `research/data/partitions/full/<cohort>/`: clinic ID indices, centralized splits, portable primary_index.csv, assignments.csv, aggregate summary/report/plot and verification JSON. These are allowlisted by `.gitignore`.
- `research/artifacts/mlruns/`: local MLflow file store; ignored. SQLite is no longer the default because the installed SQLAlchemy version was incompatible with MLflow 2.22's SQLite adapter. File-store tracking was connected and verified offline on the dev sample.

```powershell
python -m research.src.data_prep.inspect_client --config research/configs/phase1.yml --client 0
python research/scripts/prepare_data.py --profile dev
python -m pytest research/tests -q
$cohort = 'public'  # use 'all' once CheXpert is included
git add -- "research/data/partitions/full/$cohort"
git diff --cached --stat
git commit -m "add full data partitions"
```

Open `research/data/partitions/full/<cohort>/distribution.png`, `report.md` and the local client_00/label-distribution.png. Confirm source/class skew, reasonable clinic volumes, complete coverage and no known patient/pixel/client leakage. The inspect command CPU-loads every image for the selected clinic and generates its class plot. Automatic tests use the dev configuration; a local integration test checks the actual dev sample when present. Dev sampling reads existing inputs only and cannot download. A new teammate must build it after the full preparation; this laptop's sample stays intact even after raw data cleanup.

Review staged files: only ID CSVs, aggregate JSON/Markdown and the generated aggregate distribution plot belong in this commit. Original labels, patient metadata, raw paths, medical images, archives, MLflow runs, credentials and dev_sample remain ignored. `primary_index.csv` contains relative processed paths and class/source/split fields; it omits patient IDs and absolute raw paths. Share processed data on approved storage together with the matching indices; clone alone does not supply images.

Loader contract:

```python
from pathlib import Path
from research.src.data_prep.common import load_config
from research.src.data_prep.dataset import ImagingDataset
config = load_config("research/configs/phase1.yml")
clinic = ImagingDataset(config["paths"]["processed"],
                        Path(config["paths"]["partitions"]) / "client_00/train.csv")
image, label = clinic[0]  # float32 CPU [3,224,224], class 0/1/2
```

## Definition of Done status

| Gate | Status | Evidence / next owner |
|---|---|---|
| Repository skeleton and shared destination | Done locally; new commits need sharing | Existing origin points to RuralMed_FL; no Phase 2 code added. |
| Reproducible environment specs and usable environment | Done for existing environment; fresh clone/cold install handed off | Exact direct pins and Python 3.11.9/CUDA 12.6 operations verified; under-15-minute cold build not proven. |
| Data preprocessing/mapping contract | Done | Common grayscale 224 PNGs, ImageNet normalization at load, weak-label and uncertainty exclusions, original metadata retained. |
| Small local processed sample, splits, 10 clinics and client plot | Done for available sources | 300 real images, 100 per class, about 8.1 MiB. NIH 100 + COVID 200; 197 train / 45 val / 58 centralized test; clinics have 5–43 train images. Patient/pixel overlap checks passed; dev plot visually reviewed. |
| Real sample from all three sources | Blocked | No approved CheXpert files on this laptop. Its adapter is exercised with labeled synthetic test fixtures; these are never presented as real CheXpert data. Recreate the dev sample from all sources on the teammate's machine once available. |
| Full verified datasets, patient splits and 10-clinic report | Handed to teammate | Run prepare_data once on the large drive, inspect full-source skew and commit its small outputs. Full acquisition/processing was deliberately not run here. |
| Tracker created and connected | Done for dev; full run handed off | MLflow file-store experiment and configuration/data/report artifacts logged in the existing pinned environment. |
| Hosted Supabase shell and safe credentials | Blocked on team account | Template and schema sketch done; team must create its project and populate ignored .env. |
| Decisions and clean storage plan | Done | Decisions log, this handoff, dev aggregate evidence and [exact cleanup commands](phase1-storage-cleanup.md). No cleanup deletes were executed. |

Phase 1's end-to-end fresh-teammate Definition of Done remains pending the teammate's clone/build/full-data/client-plot run and CheXpert/Supabase access. COVID has no patient IDs; unknown shared patients/near-duplicates remain an explicit evaluation limitation. Do not start Phase 2 in this session.
