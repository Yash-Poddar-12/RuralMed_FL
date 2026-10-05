# Research workspace

Phase 1 prepares data and index contracts only. Follow the complete [teammate handoff](../infra/docs/phase1-handoff.md) for clone/environment setup, approved CheXpert access, storage, validation and committing outputs. The old continuation worker and launchers are retired.

## Full data on the teammate's drive

Set `RURALMED_DATA_ROOT` in the environment or ignored `.env` (see `.env.example`) to the large-drive directory. Allow about 100 GiB free, more if keeping archives. From the repo root, with the pinned Python 3.11 environment active:

```powershell
python research/scripts/prepare_data.py --profile full --delete-archives
python -m research.src.data_prep.inspect_client --config research/configs/phase1.yml --client 0
```

The single foreground command downloads pinned NIH v3 and COVID-QU-Ex v7, verifies SHA-256/CRC/counts, handles an approved local CheXpert-small release or ZIP, preprocesses all available sources, groups patients/duplicates, splits and partitions ten clinics, then logs MLflow and exports portable indices/reports. Rerun the same command after interruption. Missing CheXpert prints a pending message and creates the public cohort; adding it creates the separate all cohort. `--no-download` permits existing verified inputs only; `--no-tracking` explicitly leaves tracker verification pending. All paths derive from the repo or configured data root. See `--help` for local archive/source-root options.

Raw/processed data are under `<data-root>/{raw,processed}`. Small allowed outputs are under `research/data/partitions/full/{public,all}`; full manifests/images, archives and credentials stay ignored. The [data contract](../infra/docs/data-contract.md) defines labels, leakage rules and the normalized CPU tensor loader.

## Offline dev profile

```powershell
python research/scripts/prepare_data.py --profile dev
python -m pytest research/tests -q
python -m research.src.data_prep.inspect_client --config research/configs/phase1-dev.yml --client 0
```

Dev creation samples existing raw inputs only; it never downloads or processes full datasets. `--source-root <existing-raw-folder>` selects another existing source. The ignored `research/data/dev_sample/` holds 300 resized real images, balanced across three classes and available source strata, its manifests, tiny indices and reports (about 8.1 MiB locally). NIH has no COVID class; unavailable CheXpert is recorded explicitly. Synthetic CheXpert fixtures cover the adapter in tests. Keep the existing sample when clearing caches/raw/pilot outputs: a rerun verifies the saved PNGs and rebuilds indices without requiring raw data. To sample newly added sources on the teammate's machine, archive/move the old dev_sample first and recreate it. The dev profile lowers the per-clinic minimum to five; full remains 32. Tests always use dev settings and CPU tensors.

## Tracking

MLflow uses the ignored local file store `research/artifacts/mlruns/`, avoiding SQLite adapter dependency drift. Full preparation logs resolved config, config hash, data/split counts and reports. Optional offline dev tracking (requires installed MLflow):

```powershell
python -m research.src.data_prep.pipeline --config research/configs/phase1-dev.yml
mlflow ui --backend-store-uri research/artifacts/mlruns --host 127.0.0.1 --port 5000
```

Review the aggregate plot before any later training. Cleanup commands and measured sizes are in [the storage plan](../infra/docs/phase1-storage-cleanup.md). No federated training or Phase 2 logic is included.
