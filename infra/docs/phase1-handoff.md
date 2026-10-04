# Phase 1 status and GPU teammate handoff

Snapshot: 2026-10-05. Authoritative specification: `Phase_Docs/PHASE 1.md`. This is an interim handoff; full Phase 1 Definition of Done is still pending. No Phase 2+ logic has been implemented.

## Sections 10–11 checklist

| Deliverable | Status | Evidence / remaining prerequisite |
|---|---|---|
| Monorepo, branches, shared remote | Local setup done; remote blocked | Existing layout preserved; `main`, `dev`, `phase1-data-prep` created. Team must supply a shared remote destination. |
| Reproducible environment | Specs done; clean verification running | Python 3.11.9 and exact CUDA-enabled torch 2.7.1 / torchvision 0.22.1 pins. Hidden continuation worker completes install and CPU/native-op checks. Existing host audit uses Python 3.12 / torch 2.5.1+cu121; it is distinct from clean-install verification. |
| All three raw datasets | Partial; CheXpert blocked | COVID-QU-Ex v7 fully acquired, 85,319 CRC-verified archive members. NIH v3: 1,500 verified pilot images with matched metadata; full transfer remains resumable. CheXpert small release awaits team registration/approval. |
| Preprocessing design and label mapping | Done | `data-contract.md`, configs and adapters specify Normal / Pneumonia / COVID-19, exact weak-label proxies, uncertainty exclusions, original metadata, PNG representation and ImageNet loading normalization. |
| Processed dataset | Done for pilot; full NIH pending | All 35,420 pilot inputs decoded and resized on CPU: full 33,920 COVID images plus 1,500 NIH. 490 NIH records excluded from primary task but retained; two exact duplicates removed. Primary cohort: 34,928 unique images. |
| Patient-level splits | Done for pilot | 22,433 train / 5,539 validation / 6,956 centralized test images. Known patient and exact processed-pixel leakage audits passed. COVID has no patient IDs; supplied splits retained and unknown shared patients/near-duplicates remain a limitation. |
| Non-IID partitioning and report | Done for pilot; full-source review pending | Ten clinics; source preferences × Dirichlet alpha 0.5 × lognormal sigma 1.0. Train sizes 317–5,699 (18× range), no overlaps and complete train/validation coverage. Pilot plot visually reviewed. NIH's small pilot share limits the observable source skew; rebuild and review after full NIH/CheXpert availability. |
| Experiment tracker created and connected | Implementation done; connection pending install | Local MLflow SQLite experiment creation, config hash and artifact logging implemented in `pipeline.py`; continuation runs the actual tracked pilot after pinned environment verification. |
| Hosted Supabase project and credentials | Shell done; hosted project blocked | Minimal schema plan and `.env.example` exist; ignored `.env` has empty values. Team account/organization and project creation are required. |
| Decisions recorded continuously | Done | `decisions-log.md` records source versions, hashes, mappings, compute, split limits, partitions and passed gates as they occur. |

The Section 11 clone/build/client-load/plot Definition of Done is not fully met until the remote and clean environment verification exist. A representative client's data contract and plots are available locally; a hidden worker continues the remaining automated data gates. Supabase and CheXpert remain explicit external account/access prerequisites. No GPU is required to finish Phase 1.

## Verified pilot and report

Six data tests passed on CPU in the existing host environment. Tests cover uncertainty mapping, whole-patient grouping, cross-source exact duplicates linked through excluded records, complete client coverage, centralized test isolation, deterministic partitions, bad indices and tensor shape/dtype. Published COVID class counts matched exactly: 10,701 Normal / 11,263 Pneumonia / 11,956 COVID-19.

The aggregate [partition report](reports/phase1-representative/report.md), [summary table](reports/phase1-representative/summary.csv) and [visually reviewed distribution plot](reports/phase1-representative/distribution.png) are checked in; medical images and per-image indices stay ignored. The ordered NIH subset supports pipeline verification and must not be presented as a population-representative clinical evaluation sample.

## Current machine: continuation

Read `research/artifacts/continuation.json` for the live stage and `continuation.log` for output. `failed` includes the error; fix it before restarting. A Windows hidden worker is running `research/scripts/resume_phase1.py`. It completes the pilot, fetches the CUDA wheel with resumable ranges, installs and verifies the pinned Python environment, reruns data tests and the tracked pilot, resumes full NIH acquisition, and prepares all public data. It deliberately leaves manual visual review pending after the full report is regenerated. Avoid launching duplicate continuation workers.

Full NIH (~42 GiB compressed) takes many hours at the observed connection speed. Account details and a GPU do not speed up local transfer. CheXpert can be added later without changing the loader interface.

## GPU teammate: Phase 2 entry

1. Use `research/README.md` to install the pinned environment from `environment.yml` or `requirements.txt`; run `research/scripts/verify_environment.py` and verify the teammate's NVIDIA driver supports the CUDA 12.6 build. The script's CPU checks also work on a GPU machine.
2. Copy a completed processed directory and its matching partitions together from this machine. Pilot: `research/data/processed/representative/` and `research/data/partitions/representative/`; full public data uses the parent paths in `phase1.yml`. Read their `preprocessing.json`, `split-summary.json`, `partition.json`, `tracking.json` and configuration hash. Do not mix indices and manifests from different runs.
3. Load `ImagingDataset(processed_root, client_index_csv)`, yielding `(float32 CPU tensor [3,224,224], class ID 0/1/2)`. Move tensors to the GPU in the Phase 2 training code. Central test indices stay outside clinics. Confirm a client with `python -m research.src.data_prep.inspect_client --config research/configs/phase1-representative.yml --client 0` (omit the config argument for full data).
4. Complete the remaining Phase 1 gates and read `Phase_Docs/PHASE 2.md` before implementing the FedAvg baseline. When CheXpert access arrives, enable its adapter, select fresh processed/partition directories, rerun preparation, and retain old config/data hashes for comparison.

Shared remote and hosted Supabase project must be created/provided in the team's accounts. Fill actual credentials directly in ignored `.env`; never put a service role key in Git or frontend code.
