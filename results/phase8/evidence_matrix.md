# Phase 8 Evidence Matrix

This matrix is based only on result files and reports already present in the
repository. It distinguishes controlled ablations from supporting evidence so
the faculty review does not overstate what was measured.

| Phase 8 question | Existing evidence | Status | Faculty-safe conclusion |
|---|---|---|---|
| IID vs. non-IID partitioning | Phase 1 non-IID partition reports only | Not controlled | The project uses heterogeneous non-IID clinics; an IID counterfactual is not present. |
| Error feedback on vs. off | Phase 3 report documents Top-K residual error feedback and stable compression results | Not controlled | Error feedback is part of the implemented compression method; no off-switch comparison is available. |
| Staleness weighting on vs. off | Phase 4 FedBuff results and buffer-size sweep | Not controlled | FedBuff with staleness-aware aggregation was evaluated; a uniform-weighting counterfactual is not present. |
| Simple dropout vs. realistic connectivity | Phase 4 dropout figures and Phase 5 Markov connectivity report | Supporting comparison | The project progresses from probability dropout to bandwidth- and state-driven connectivity, but the attached results are not a perfectly matched controlled pair. |
| Compression under constrained connectivity | Phase 5 256 kbps interaction results and Phase 6 joint results | Supported | Top-K 1% retains 0.9256 macro-AUROC, 0.8432 macro-F1, and 99.0% COVID sensitivity at 256 kbps; uncompressed reaches 0.9082 AUROC, 0.7842 F1, and 76.1% sensitivity. |
| FedBuff vs. synchronous dropout | Phase 4 comparison figures and Phase 6 async-minus-sync output | Supported with one seed | The comparison exists, but Phase 6 contains one seed per point, so it should be presented as an observed result rather than a statistically general conclusion. |
| Full-system operating point | Phase 6 recommendation and heatmaps | Supported with one seed | FedBuff + Top-K 1% + 256 kbps + 50% dropout is the current practical operating point. |

## Evidence locations

- Phase 1 partition evidence: `infra/docs/reports/phase1-representative/`
- Phase 3 compression evidence: `infra/docs/reports/phase3/report.md`
- Phase 5 connectivity evidence: `infra/docs/reports/phase5/report.md`
- Phase 6 raw runs: `results/phase6/raw/`
- Phase 6 aggregate metrics: `results/phase6/summary.csv`
- Phase 6 operating point: `results/phase6/recommended_operating_point.md`
- Phase 6 figures: `results/phase6/accuracy_heatmap.png`, `results/phase6/auroc_heatmap.png`, and `results/phase6/async_heatmap.png`

## Scope statement

The repository contains eight Phase 6 result JSON files, all with seed 42.
The Phase 6 configuration plans 51 runs and three seeds per core condition,
so the attached data should be described as a completed implementation and
representative sweep, not as a fully replicated 51-run statistical study.