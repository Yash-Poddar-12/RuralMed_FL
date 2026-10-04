# Phase 1 — Foundation: Environment Setup, Repository Structure & Dataset Preparation

**Project:** Communication-Efficient Federated Learning for Diagnostic Imaging under Intermittent Rural Connectivity
**Team:** Pavan D Umesh (23BDS0046) · Aayush Sood (23BDS0177) · Yash Poddar (23BDS0195)
**Phase:** 1 of 8
**Estimated Duration:** 1.5–2 weeks

---

## 1. Where This Fits in the Overall Project

| Phase | Name | Depends on |
|---|---|---|
| **1** | **Foundation: Env, Repo, Data** | — (this file) |
| 2 | FL Core Baseline (FedAvg) | Phase 1 |
| 3 | Communication-Efficient Compression | Phase 2 |
| 4 | Straggler-Tolerant Async Aggregation | Phase 2 |
| 5 | Intermittent Connectivity Engine | Phase 4 |
| 6 | Evaluation & Experiment Orchestration | Phases 3–5 |
| 7 | Monitoring Dashboard (Next.js/Express/Supabase) | Phase 6 (light bootstrap starts here) |
| 8 | Integration, Ablations & Final Report | All |

Nothing in later phases can start reliably until Phase 1's three outputs exist: a **working repo skeleton**, a **reproducible environment**, and **preprocessed, partitioned datasets sitting on disk in a known format**. This phase produces no FL logic yet — it produces the ground every later phase stands on.

---

## 2. Objectives of Phase 1

By the end of this phase you should be able to answer "yes" to all of the following:

- [ ] Any team member can clone the repo and get an identical working environment in under 15 minutes.
- [ ] All three datasets (or representative subsets) are downloaded, verified, and preprocessed into a common format.
- [ ] A documented, justified non-IID partitioning scheme exists that turns the pooled data into N simulated "rural clinics."
- [ ] A Supabase project exists (empty schema is fine) so Phase 7 doesn't start from zero.
- [ ] Every design decision (label taxonomy, client count, split ratios) is written down in a decisions log — not just decided verbally.

---

## 3. Repository & Project Structure

Use a **monorepo** — one repo, two clearly separated top-level concerns. This avoids sync pain between the ML research code and the web dashboard later, since experiment IDs, metric schemas, and config files need to be shared between them.

```
fl-rural-imaging/
├── research/                      # All Python / ML work (Phases 1-6, 8)
│   ├── data/
│   │   ├── raw/                   # Untouched downloaded datasets (gitignored)
│   │   ├── processed/             # Resized, normalized, harmonized images + labels
│   │   └── partitions/            # Per-client index files (which images belong to which "clinic")
│   ├── configs/                   # YAML configs (dataset paths, partition params, seeds)
│   ├── notebooks/                 # Exploratory analysis only — nothing load-bearing lives here
│   ├── src/
│   │   ├── data_prep/             # Download, verify, preprocess, harmonize scripts
│   │   ├── partitioning/          # Non-IID client-splitting logic
│   │   ├── fl_core/                # (Phase 2+) FedAvg, compression, aggregation
│   │   ├── simulation/            # (Phase 5+) connectivity/dropout trace generation
│   │   └── eval/                  # (Phase 6+) metrics, experiment runner
│   ├── environment.yml            # Conda environment spec
│   ├── requirements.txt           # Pip fallback
│   └── README.md
│
├── web/                            # Dashboard (Phase 7)
│   ├── frontend/                  # Next.js app
│   └── backend/                   # Node.js + Express API
│
├── infra/
│   ├── supabase/                  # SQL migrations / schema definitions
│   └── docs/                      # Decisions log, architecture notes, diagrams
│
├── .gitignore                     # Must exclude research/data/raw and research/data/processed
└── README.md                       # Top-level project overview
```

**Why this shape:**
- `research/` and `web/` never need to be deployed together, but they need to agree on shared vocabulary (experiment IDs, metric names) — that shared contract lives in `infra/docs/`.
- Raw and processed medical imaging data must **never** be committed to git (size + potential licensing terms) — enforce this in `.gitignore` from day one, not after someone accidentally commits 40GB.
- `configs/` as YAML (not hardcoded values) means Phase 6's experiment orchestration can sweep over partition schemes, compression settings, etc. without touching code.

---

## 4. Environment Setup

### 4.1 Python / ML Environment

| Component | Choice | Why |
|---|---|---|
| Python version | 3.10 or 3.11 | Stable compatibility with Flower, PyTorch, torchvision as of your dataset's release windows |
| Environment manager | Conda (or `venv` + pip if disk-constrained) | Easier CUDA/toolkit version pinning for imaging workloads |
| Deep learning framework | PyTorch + torchvision | Best-supported by Flower/FedML tutorials for medical imaging FL |
| FL framework | **Flower (`flwr`)** | Lighter-weight than FedML, framework-agnostic, large community, easiest to extend with custom aggregation strategies (needed for Phases 3–4) |
| Experiment tracking | Weights & Biases (or MLflow if offline-only is preferred) | You'll be running dozens of experiment configs later (bandwidth × dropout × compression) — you need this from Phase 1, not bolted on later |
| Image handling | Pillow, OpenCV (optional), `albumentations` for augmentation | Standard for chest X-ray preprocessing |
| Numerics/analysis | numpy, pandas, scikit-learn, matplotlib/seaborn | Partitioning stats, class-imbalance analysis, plots for the eventual report |

**Decision point to record in the decisions log:** whether to develop locally (if any team member has a CUDA GPU) or on Google Colab Pro / Kaggle Notebooks (free T4 GPUs, but session time limits matter for later long FL simulation runs). Given three medical imaging datasets and dozens of later FL rounds, **budget for at least one paid GPU tier** (Colab Pro, Kaggle's free 30hrs/week, or a university compute cluster if VIT provides one) — CPU-only federated simulation across 3 large datasets will not finish in usable time.

### 4.2 Web/Dashboard Toolchain (bootstrap only — full build is Phase 7)

Even though the dashboard isn't built until Phase 7, install and register these now so environment drift doesn't happen later:

| Component | Choice |
|---|---|
| Frontend | Next.js (latest stable) |
| Backend API | Node.js + Express |
| Database | Supabase (PostgreSQL) |
| Node version | LTS (pin via `.nvmrc`) |

### 4.3 Version Control

- Git repo initialized with the structure above.
- Branch strategy: `main` (stable), `dev` (integration), short-lived feature branches per phase/task (e.g. `phase1-data-prep`).
- Commit discipline: no raw or processed data, no API keys/Supabase credentials — use `.env` files (gitignored) with an `.env.example` template checked in instead.

---

## 5. Dataset Acquisition

| Dataset | Source | Size (approx.) | Access Notes |
|---|---|---|---|
| **NIH ChestX-ray14** | NIH Clinical Center (via Kaggle mirror or official NIH box link) | ~42GB, 112,120 images, 14 disease labels | Publicly downloadable, no registration required |
| **CheXpert** | Stanford ML Group | ~11GB (small) / ~440GB (full-res) — use the small/downsampled release | Requires filling a Stanford data-use registration form (free, but takes a day or two for approval — **start this download request in the first 48 hours of Phase 1**, it's the most likely bottleneck) |
| **COVID-QU-Ex** | Qatar University (via Kaggle) | ~2–3GB, ~33,900 images across COVID-19 / Non-COVID pneumonia / Normal | Kaggle account required, direct download |

**Tasks:**
1. Register/apply for CheXpert access immediately (longest lead time).
2. Download NIH ChestX-ray14 and COVID-QU-Ex in parallel.
3. Verify each dataset's integrity (checksum or file-count validation against the published dataset card) before touching the data — a partial/corrupt download silently poisons every later phase.
4. Store all raw data under `research/data/raw/<dataset_name>/`, untouched, as the immutable source of truth.
5. Log dataset versions/download dates in `infra/docs/decisions-log.md` — dataset releases get updated over time, and reproducibility later depends on knowing exactly what you trained on.

**Storage decision:** given combined raw size will likely exceed 50GB, decide now whether this lives on a personal machine, an external drive, university compute storage, or a cloud bucket. Do **not** rely on Google Drive as the working directory for active preprocessing — I/O latency will make preprocessing painfully slow. Drive/cloud storage is fine as a backup only.

---

## 6. Data Preprocessing Pipeline (Design, No Code Yet)

### 6.1 Label Harmonization — the key design decision of this phase

The three datasets do **not** share a label taxonomy out of the box:

- NIH ChestX-ray14: 14 multi-label disease findings (e.g., Atelectasis, Effusion, Pneumonia, No Finding, ...)
- CheXpert: 14 multi-label findings, overlapping but not identical to NIH's set, plus uncertainty labels (`-1`)
- COVID-QU-Ex: 3-class single-label (COVID-19, Non-COVID Pneumonia, Normal)

**Recommended unified taxonomy for this project:** a 3-class task — `Normal`, `Pneumonia` (bacterial/viral/other), `COVID-19` — mapped as follows:
- COVID-QU-Ex maps directly (it's already in this taxonomy).
- NIH and CheXpert: `No Finding` → `Normal`; `Pneumonia` (and closely related infiltrate/consolidation labels, documented explicitly) → `Pneumonia`; everything else is **excluded from the primary task** (kept in processed data as metadata, so it's not lost if the taxonomy is revisited later).
- CheXpert uncertainty labels (`-1`): treated as "exclude from that specific label" rather than guessed — do not silently convert uncertain labels to positive or negative.

This keeps the classification task well-defined and clinically meaningful (COVID/pneumonia triage is exactly the kind of decision a bandwidth-constrained rural clinic would need), while naturally creating **non-IID structure across datasets** — which Section 7 exploits.

*Record this mapping decision, and any label exclusions, explicitly in the decisions log — it needs to be citable in the final report's methodology section.*

### 6.2 Image Preprocessing Standard

- Resize all images to a consistent resolution (224×224 is the common baseline for CNN backbones like ResNet/DenseNet used in the cited literature).
- Convert to a single channel or replicate to 3-channel depending on backbone choice (document the choice).
- Normalize using either dataset-wide computed mean/std or standard ImageNet statistics if using ImageNet-pretrained backbones (recommended for faster convergence given federated training is already sample- and round-constrained).
- Store processed images either as re-saved image files or as a serialized tensor format (e.g., `.pt` shards or `.npz`) — decide based on whichever loads faster during FL simulation (many FL rounds will re-read this data repeatedly, so load speed matters more than usual).

### 6.3 Train/Val/Test Split Strategy

- Split **before** partitioning into clients, at the patient level, not the image level, wherever patient IDs are available (both NIH and CheXpert provide these) — this avoids the same patient's images leaking across train/test, which would inflate reported accuracy.
- Recommended split: 70% train / 15% validation / 15% test, stratified by class where feasible.
- The **test set stays centralized** (used only for final global-model evaluation) — it is never partitioned into simulated clients.

---

## 7. Non-IID Partitioning Strategy — Simulating Rural Clinics

This is the piece that makes the project's "non-IID diagnostic imaging" claim concrete and defensible, so it deserves a deliberate design rather than an arbitrary random split.

### 7.1 What a "client" represents

Each simulated FL client = one rural clinic. Recommended: **8–15 clients total**, combining two sources of non-IID-ness rather than one, since real rural clinics would differ in both *what equipment/patients they see* and *how much data they have*:

1. **Natural non-IID via dataset source:** clients drawing primarily from COVID-QU-Ex will have a very different label distribution than clients drawing from NIH/CheXpert — this mirrors real regional differences (e.g., a clinic during a COVID surge vs. a general-purpose rural clinic).
2. **Synthetic label-skew via Dirichlet partitioning:** within each dataset's contribution, apply a Dirichlet distribution (a standard, citable non-IID partitioning method used across the FL literature you reviewed) to control how skewed each client's label distribution is, with a tunable concentration parameter — lower values produce more extreme, more "rural-realistic" skew.

### 7.2 Data volume imbalance

Real rural clinics don't just differ in label mix — they differ wildly in patient volume. Assign client dataset sizes using a long-tailed distribution (e.g., a power-law or lognormal draw) rather than splitting evenly, so some simulated clients have a handful of images and others have thousands. This directly sets up Phase 4/5's straggler and dropout simulation to be realistic rather than artificial.

### 7.3 What to produce and store

For each client, store an index file (not a copy of the data) listing which processed-image IDs belong to that client. This keeps `research/data/partitions/` small and lets you regenerate alternate partitioning schemes later (e.g., for an ablation comparing IID vs. non-IID performance in Phase 8) without re-touching the processed images.

Also produce a **partition summary report** (a table/plot of each client's size and label distribution) — this becomes a figure in the final report and a sanity check that the partitioning actually behaves as intended before any FL code is written on top of it.

---

## 8. Experiment Tracking & Config Management

- Every later experiment (compression method × dropout rate × bandwidth cap) needs a unique, logged configuration — set this discipline up now, not once you already have a mess of ad hoc results.
- Use YAML config files under `research/configs/` for: dataset paths, preprocessing parameters, partition scheme + random seed, and (from Phase 2 onward) FL hyperparameters.
- Connect Weights & Biases (or MLflow) at the project level now, even though there's nothing to log yet beyond dataset statistics — this avoids retrofitting tracking into Phase 2's training loop under time pressure.

---

## 9. Supabase Project Bootstrap (light touch)

Full dashboard integration is Phase 7, but create the project shell now so later phases (especially Phase 6's experiment orchestration, which will want to write results somewhere) aren't blocked:

- [ ] Create a Supabase project (free tier is sufficient at this stage).
- [ ] Record project URL and API keys in `.env.example` (placeholder values) and a real `.env` (gitignored).
- [ ] Sketch — do not fully build — a minimal schema plan in `infra/supabase/` covering: `experiments`, `clients`, `training_rounds`, `metrics`. Table creation itself happens in Phase 7 once the metrics that actually need storing are known from Phases 2–6.

---

## 10. Deliverables Checklist

- [ ] Monorepo initialized with the structure in Section 3, pushed to a shared remote (GitHub/GitLab).
- [ ] `environment.yml`/`requirements.txt` that a teammate can use to reproduce the environment from scratch.
- [ ] All three datasets downloaded, verified, and sitting under `research/data/raw/`.
- [ ] Preprocessing pipeline design finalized and documented (Section 6), including the label harmonization mapping table.
- [ ] Processed dataset under `research/data/processed/` in the agreed format.
- [ ] Patient-level train/val/test split completed and saved as index files.
- [ ] Non-IID client partitioning completed (8–15 clients) with a partition summary report (table + distribution plots).
- [ ] Experiment tracking project created and connected.
- [ ] Supabase project created with credentials stored safely.
- [ ] `infra/docs/decisions-log.md` containing every decision made in this phase with a one-line justification each.

---

## 11. Definition of Done

Phase 1 is complete when a teammate who was *not* involved in setup can: clone the repo, build the environment from the spec file, point it at the processed data location, load any single client's partition, and see a label-distribution plot for that client — all without needing to ask a question in the group chat.

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| CheXpert access approval delays the whole phase | Apply for access on day 1; start NIH/COVID-QU-Ex preprocessing in parallel so the team isn't blocked |
| Combined dataset size exceeds available local storage | Decide cloud/external storage strategy before downloading, not after a failed download |
| Label harmonization choices turn out to be wrong once FL training starts | Keep original per-dataset labels in the processed metadata (don't discard them) so the taxonomy can be revisited without re-downloading/re-preprocessing |
| Partitioning scheme produces clients too small to train on | Generate the partition summary report (Section 7.3) and visually sanity-check before moving to Phase 2 — don't discover this mid-training |
| Team members end up on inconsistent library versions | Pin exact versions in `environment.yml`, not `>=` ranges |

---

## 13. Next Phase Preview

**Phase 2 — FL Core Baseline** will use the partitioned data from this phase to stand up a working Flower-based FedAvg simulation (matching the McMahan et al. baseline from your literature review) across all simulated clients, with no compression or connectivity constraints yet — establishing the accuracy ceiling that Phases 3–5's efficiency techniques will be measured against.
