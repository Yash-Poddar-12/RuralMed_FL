# Communication-Efficient Federated Learning for Diagnostic Imaging

University capstone project investigating three-class chest-radiograph federated learning under heterogeneous rural-clinic data, limited bandwidth, intermittent connectivity, and client dropout.

The project simulates ten clinics that keep their training images locally. Each clinic fine-tunes a DenseNet-121 model, sends an update to a server, and receives the aggregated global model. The study measures whether compression and dropout-tolerant aggregation can preserve diagnostic performance when network conditions are unreliable.

For the complete beginner-friendly explanation, see [explanation.md](explanation.md).

## Team

- Yash Poddar (23BDS0195)
- Pavan D Umesh (23BDS0046)
- Aayush Sood (23BDS0177)

## Research question

Can federated chest-X-ray classification remain useful when simulated rural clinics have different data distributions, unequal data volumes, slow links, and temporary disconnections?

The primary task harmonizes public data into three classes:

- `Normal`
- `Pneumonia`
- `COVID-19`

The main techniques are:

- sample-count-weighted FedAvg as the uncompressed baseline;
- Top-K update sparsification with client-side error feedback;
- QSGD-style stochastic quantization;
- FedBuff-style buffered aggregation with staleness weighting;
- Markov-state connectivity profiles that convert payload size and bandwidth into transmission delays and dropouts.

## Phase overview

| Phase | Contribution |
|---|---|
| 1 | Environment, data acquisition, label harmonization, leakage-safe splits, and non-IID clinic partitions |
| 2 | DenseNet-121 Flower simulation and uncompressed FedAvg baseline |
| 3 | Top-K sparsification, error feedback, QSGD quantization, and exact communication accounting |
| 4 | Buffered asynchronous aggregation, staleness weighting, dropout modeling, and sync comparison |
| 5 | Heterogeneous Markov connectivity, bandwidth caps, transmission-time simulation, and uptime diagnostics |
| 6 | Config-driven joint sweeps, result normalization, heatmaps, operating-point selection, and Supabase export |
| 7 | Next.js and Express dashboard for experiments, runs, heatmaps, convergence, and clinic events |
| 8 | Integration evidence, planned ablations, limitations, report, and demo packaging |

The phase specifications are in [Phase_Docs](Phase_Docs/). Current evidence and limitations are summarized in [results/phase8/evidence_matrix.md](results/phase8/evidence_matrix.md) and [results/phase8/faculty_review_brief.md](results/phase8/faculty_review_brief.md).

## Repository layout

- `research/`: Python data preparation, partitioning, federated-learning core, simulation, evaluation, configs, scripts, and tests.
- `web/`: Next.js frontend and Node.js/Express backend for the Phase 7 dashboard.
- `infra/`: Supabase migration, data contracts, decisions, handoffs, and generated phase reports/figures.
- `Phase_Docs/`: detailed specifications for Phases 1 through 8.
- `results/`: checked-in Phase 6 summaries/raw result references and Phase 8 evidence artifacts.
- `explanation.md`: full project explanation for readers unfamiliar with federated learning or this codebase.

Important implementation surfaces:

| File or directory | Role |
|---|---|
| `research/src/data_prep/` | Dataset adapters, preprocessing, splitting, validation, and image loading |
| `research/src/partitioning/partition.py` | Source-skew, Dirichlet label-skew, and long-tailed client allocation |
| `research/src/fl_core/client.py` | Local clinic training and optional update compression |
| `research/src/fl_core/server.py` | Flower FedAvg strategy, centralized test evaluation, and tracking |
| `research/src/fl_core/compression/` | Top-K, QSGD, residuals, and wire-byte accounting |
| `research/src/fl_core/aggregation/` | FedBuff and synchronous-with-dropout strategies |
| `research/src/simulation/` | Connectivity states, client profiles, Markov transitions, and delay adapter |
| `research/src/eval/` | Phase 6 sweep runner, summaries, heatmaps, and Supabase batch export |
| `research/scripts/run_phase5.py` | Main Phase 5/6-compatible training entry point |
| `web/backend/server.js` | Dashboard REST API |
| `web/frontend/app/` | Overview, experiment, run, heatmap, and clinic views |

## Current checked-in results

The current Phase 6 result directory contains eight raw runs, all with seed 42. The configured study plans a larger three-seed sweep, so these artifacts should be described as representative evidence rather than a fully replicated statistical study.

The current operating-point artifact selects:

```text
FedBuff + Top-K 1% + 256 kbps + 50% dropout
Accuracy:     0.8167
Macro-AUROC:  0.9256
Macro-F1:     0.8432
Seeds:        1
```

The strongest Phase 5 interaction report at 256 kbps compares Top-K 1%, Top-K 10%, and uncompressed updates. It reports Top-K 1% at macro-AUROC `0.9256`, macro-F1 `0.8432`, and COVID-19 sensitivity `99.0%`, while the uncompressed configuration reports `0.9082`, `0.7842`, and `76.1%`, respectively. These are simulated results, not clinical deployment claims.

The checked-in evidence does not include controlled result files for every planned Phase 8 ablation, including IID-vs-non-IID, error-feedback-off, and staleness-weighting-off comparisons. Do not present those as completed causal studies.

## Data and storage policy

Raw and processed medical-imaging data, model artifacts, credentials, and local experiment stores are ignored by Git. Full acquisition and processing belong on the teammate's large drive. Do not download full datasets, process full datasets, or start background workers on this laptop.

Local development uses the ignored approximately 300-image sample. Full preparation is performed through the foreground `prepare_data.py` workflow after configuring `RURALMED_DATA_ROOT`. CheXpert access and local files may require separate registration and approval.

Start with [Phase 1 status and GPU handoff](infra/docs/phase1-handoff.md), [research commands](research/README.md), the [Phase 8 handoff runbook](PHASE8_HANDOFF_RUNBOOK.md), and the [decisions log](infra/docs/decisions-log.md).

## Environment and commands

The research code is Python-based and expects the project virtual environment or the environment described by [research/environment.yml](research/environment.yml) and [research/requirements.txt](research/requirements.txt). Full FL runs expect CUDA-capable hardware because the client currently aborts when CUDA is unavailable. Data preparation tests and most unit tests are designed to run offline on CPU.

Useful commands from the repository root:

```bash
# Verify the Python environment without downloading data
.venv/bin/python research/scripts/verify_environment.py

# Run the offline test suite when pytest is installed
.venv/bin/python -m pytest -q research/tests

# Prepare or verify the local development sample
.venv/bin/python research/scripts/prepare_data.py --profile dev

# Generate Phase 6 summaries from existing raw results
.venv/bin/python -m research.src.eval.joint_metric_plots \
	--results-dir results/phase6/raw \
	--output-dir results/phase6

# Generate the Supabase batch payload without ingesting it
.venv/bin/python -m research.src.eval.supabase_writer \
	--results-dir results/phase6/raw \
	--batch-json results/phase6/supabase_batch.json
```

The Phase 6 PowerShell launcher is retained for the original Windows workflow in [research/scripts/run_phase6_all.ps1](research/scripts/run_phase6_all.ps1). On macOS/Linux, invoke the Python modules directly or use the project-specific runbook instructions.

## Dashboard

The dashboard is a read-only visualization layer over Phase 6 results:

```text
Python result JSON -> Supabase batch rows -> Express API -> Next.js frontend
```

The backend reads `SUPABASE_URL` and either `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_ANON_KEY` from the environment. The frontend reads `NEXT_PUBLIC_API_URL`, defaulting to `http://localhost:4000`. The backend exposes experiments, runs, round metrics, client events, joint metrics, and client metadata. It does not run federated training.

See [web/README.md](web/README.md) for web-specific setup.

## Reproducibility and scope

The repository uses YAML configuration, fixed seeds, manifests, configuration hashes, local MLflow tracking, raw JSON results, generated summaries, and Supabase row contracts to preserve provenance.

The project is a reproducible simulation study of communication-aware federated learning. Connectivity is modeled rather than measured from deployed clinics; the label taxonomy is simplified; and the current Phase 6 artifacts have one seed per populated point. These boundaries are part of the result and should remain explicit in reports and presentations.
