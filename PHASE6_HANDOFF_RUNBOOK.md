# Phase 6 Handoff Runbook

This document explains how to run Phase 6 when the project is handed over
directly at Phase 6.

Phase 6 uses the existing Phase 3–5 implementation through the Phase 5
simulation entry point. You do **not** need to manually rerun the Phase 2,
Phase 3, Phase 4, or Phase 5 experiment commands. You only need the prepared
data, the Python environment, and the Phase 6 commands below.

## 1. What Phase 6 does

Phase 6 runs the joint evaluation across:

- Compression settings:
  - Uncompressed baseline
  - Top-K 1%
  - Top-K 10%
  - QSGD 4-bit
- Connectivity conditions:
  - No dropout
  - Moderate dropout
  - Severe dropout
  - Extreme dropout
- Aggregation:
  - FedBuff buffered asynchronous aggregation
  - Synchronous dropout comparison
- Random seeds:
  - 42
  - 43
  - 44

The default configuration expands to 51 runs:

- 48 core FedBuff runs
- 3 synchronous comparison runs

Each run writes raw per-seed results. Phase 6 then aggregates those results,
generates heatmaps, calculates mean and standard deviation, identifies an
operating point, and prepares Supabase rows.

## 2. Repository entry points

Run all commands from the repository root.

| Purpose | File |
|---|---|
| Phase 6 sweep definition | `research/configs/phase6_core_joint_sweep.yaml` |
| Sweep runner | `research/src/eval/experiment_runner.py` |
| Existing training simulation used by the runner | `research/scripts/run_phase5.py` |
| Joint metric summaries and heatmaps | `research/src/eval/joint_metric_plots.py` |
| Supabase export and ingestion | `research/src/eval/supabase_writer.py` |
| Supabase schema migration | `infra/supabase/migrations/0001_experiments_runs_rounds_clients_schema.sql` |
| Phase 6 tests | `research/tests/test_phase6.py` |

## 3. Prerequisites

### 3.1 Python environment

The project uses Python 3.11 and the pinned packages in
`research/environment.yml` and `research/requirements.txt`.

Using Conda:

```bash
conda env create -f research/environment.yml
conda activate fl-rural-imaging
```

If the environment already exists:

```bash
conda activate fl-rural-imaging
```

Check the installation:

```bash
python -m pip check
python research/scripts/verify_environment.py
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The existing DenseNet client requires CUDA for training. The final Phase 6
simulation should therefore run on the project GPU machine. A CPU-only
machine can be used for configuration checks and synthetic/reporting checks,
but not for the full training sweep.

### 3.2 Prepared dataset

The preferred handoff is that the previous teammate supplies the prepared
Phase 1 data. You do not need to download or preprocess it again if these
directories already exist:

```text
<data-root>/processed/full/public/
research/data/partitions/full/public/
```

The processed data directory must contain at least:

```text
<data-root>/processed/full/public/
├── primary_manifest.csv
└── splits/
    └── test.csv
```

The partition directory must contain:

```text
research/data/partitions/full/public/
├── client_00/
│   ├── train.csv
│   └── val.csv
├── client_01/
├── ...
├── client_09/
└── splits/
    └── test.csv
```

Set the data root on macOS/Linux:

```bash
export RURALMED_DATA_ROOT="/path/to/large-drive/ruralmed-data"
```

You can also put the value in the ignored `.env` file:

```text
RURALMED_DATA_ROOT=/path/to/large-drive/ruralmed-data
```

Verify the essential files:

```bash
test -f "$RURALMED_DATA_ROOT/processed/full/public/primary_manifest.csv"
test -f "research/data/partitions/full/public/splits/test.csv"
test -f "research/data/partitions/full/public/client_00/train.csv"
test -f "research/data/partitions/full/public/client_09/train.csv"
```

If the commands above produce no error, the minimum data layout is present.

## 4. If the dataset was not handed over

If the prepared data is unavailable, run the one-time Phase 1 preparation
command. This does **not** run the Phase 2–5 experiment suites; it only creates
the data required by Phase 6.

Prepare the full public cohort:

```bash
cd /path/to/RuralMed_FL
cp .env.example .env
export RURALMED_DATA_ROOT="/path/to/large-drive/ruralmed-data"

python research/scripts/prepare_data.py \
  --profile full \
  --delete-archives
```

The full preparation requires substantial storage. Reserve approximately
125–150 GB for a fresh run, depending on whether archives are retained.

CheXpert is optional and must be obtained through the team's approved
CheXpert-small access. The preparation command does not bypass access
requirements. If CheXpert is unavailable, the public NIH/COVID cohort is still
usable for Phase 6.

After preparation, inspect one client:

```bash
python -m research.src.data_prep.inspect_client \
  --config research/configs/phase1.yml \
  --client 0
```

## 5. Configure the Phase 6 data profile

The default Phase 6 config uses:

```yaml
phase1_config: research/configs/phase1.yml
```

This is the full-data configuration.

For a local dev-sample smoke test only, change it to:

```yaml
phase1_config: research/configs/phase1-dev.yml
```

Then prepare the dev sample if it is not already present:

```bash
python research/scripts/prepare_data.py --profile dev
```

The dev sample is for pipeline validation and is not a replacement for the
final full-data Phase 6 results.

## 6. Validate the Phase 6 sweep without training

From the repository root:

```bash
python -m research.src.eval.experiment_runner \
  --config research/configs/phase6_core_joint_sweep.yaml \
  --output-dir results/phase6/raw \
  --dry-run
```

Expected result:

```text
planned_runs: 51
```

This command only expands the configuration. It does not start model
training.

## 7. Run a one-run smoke test

Before starting all 51 GPU runs, launch one run:

```bash
python -m research.src.eval.experiment_runner \
  --config research/configs/phase6_core_joint_sweep.yaml \
  --output-dir results/phase6/smoke \
  --max-runs 1
```

Check that the output contains:

```text
results/phase6/smoke/manifest.json
results/phase6/smoke/*.json
results/phase6/smoke/*.log
```

If the smoke test fails, inspect the corresponding `.log` file before
starting the complete sweep.

## 8. Run the complete Phase 6 sweep

```bash
python -m research.src.eval.experiment_runner \
  --config research/configs/phase6_core_joint_sweep.yaml \
  --output-dir results/phase6/raw
```

The runner launches the individual configurations sequentially. Each run
retains:

- Seed
- Compression configuration
- Aggregation strategylong time because every configuration performs actual
federated DenseNet training. Keep the GPU machine awake and connected.

## 9. Generate Phase 6 summaries and plots

After all runs complete:

```bash
python -m research.src.eval.joint_metric_plots \
  --results-dir results/phase6/raw \
  --output-dir results/phase6
```

Expected outputs:

```text
results/phase6/
├── accuracy_heatmap.png
├── auroc_heatmap.png
├── summary.csv
├── recommended_operating_point.json
└── recommended_operating_point.md
```

If synchronous comparison results are present, the runner also writes:

```text
results/phase6/async_minus_sync.csv
results/phase6/async_heatmap.png
```

The summary uses the three seeds to calculate mean and standard deviation for
the headline metrics:

- Test accuracy
- Macro-AUROC
- Macro-F1

The recommended operating point is selected from compressed configurations
with at least three seeds and is documented in
`recommended_operating_point.md`.

## 10. Export results for Supabase

First create a local batch payload:

```bash
python -m research.src.eval.supabase_writer \
  --results-dir results/phase6/raw \
  --batch-json results/phase6/supabase_batch.json
```

This export does not require Supabase credentials.

## 11. Apply the Supabase migration

Apply this SQL file in the Supabase SQL Editor or using the team's migration
workflow:

```text
infra/supabase/migrations/0001_experiments_runs_rounds_clients_schema.sql
```

The migration creates:

- `experiments`
- `runs`
- `rounds`
- `clients`
- `client_round_events`

The schema is normalized for the Phase 7 dashboard.

## 12. Ingest results into Supabase

Set the server-only credentials in the shell:

```bash
export SUPABASE_URL="https://YOUR_PROJECT.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="YOUR_SERVER_ONLY_SERVICE_ROLE_KEY"
```

Then ingest:

```bash
python -m research.src.eval.supabase_writer \
  --results-dir results/phase6/raw \
  --batch-json results/phase6/supabase_batch.json \
  --ingest
```

Never commit `.env`, the service-role key, or any other Supabase secret.
Never expose the service-role key to the frontend.

## 13. Recommended complete command sequence

If the prepared dataset was handed over, the complete sequence is:

```bash
cd /path/to/RuralMed_FL
conda activate fl-rural-imaging

export RURALMED_DATA_ROOT="/path/to/large-drive/ruralmed-data"

python -m research.src.eval.experiment_runner \
  --config research/configs/phase6_core_joint_sweep.yaml \
  --output-dir results/phase6/raw \
  --dry-run

python -m research.src.eval.experiment_runner \
  --config research/configs/phase6_core_joint_sweep.yaml \
  --output-dir results/phase6/smoke \
  --max-runs 1

python -m research.src.eval.experiment_runner \
  --config research/configs/phase6_core_joint_sweep.yaml \
  --output-dir results/phase6/raw

python -m research.src.eval.joint_metric_plots \
  --results-dir results/phase6/raw \
  --output-dir results/phase6

python -m research.src.eval.supabase_writer \
  --results-dir results/phase6/raw \
  --batch-json results/phase6/supabase_batch.json
```

After the Supabase migration is applied and credentials are configured:

```bash
python -m research.src.eval.supabase_writer \
  --results-dir results/phase6/raw \
  --batch-json results/phase6/supabase_batch.json \
  --ingest
```

## 14. Validation and troubleshooting

Run the Phase 6 tests inside the project environment:

```bash
python -m pytest research/tests/test_phase6.py -q
```

Common issues:

### Missing partition files

Error symptoms include missing `client_00/train.csv` or missing
`splits/test.csv`.

Resolution:

- Ask the previous teammate for `research/data/partitions/full/public/`.
- Confirm `RURALMED_DATA_ROOT` points to the directory containing
  `processed/full/public/`.
- If the data was never prepared, run the full preparation command in
  Section 4.

### CUDA unavailable

Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

The existing client intentionally rejects CPU-only training. Use the team's
GPU machine with compatible NVIDIA drivers and the pinned CUDA PyTorch wheel.

### A run fails

Inspect the run log:

```bash
less results/phase6/raw/<run-name>.log
```

Do not delete successful raw JSON files. Fix the environment or configuration
and rerun into a new output directory, or rerun only the affected
configuration after preserving the existing results.

### Supabase ingestion fails

Check:

- The migration was applied successfully.
- `SUPABASE_URL` is correct.
- `SUPABASE_SERVICE_ROLE_KEY` is valid and server-side only.
- The batch JSON was generated from completed result files.

## 15. What you do not need to run

You do not need to manually run:

- Phase 2 FedAvg baseline experiments
- Phase 3 compression sweep experiments
- Phase 4 async/dropout sweep experiments
- Phase 5 bandwidth sweep experiments

Phase 6 invokes the existing Phase 5 simulation entry point with the required
compression, aggregation, bandwidth, dropout, and seed overrides.
