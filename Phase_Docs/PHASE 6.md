# Phase 6 — Evaluation & Experiment Orchestration: The Joint Metric

**Project:** Communication-Efficient Federated Learning for Diagnostic Imaging under Intermittent Rural Connectivity
**Team:** Pavan D Umesh (23BDS0046) · Aayush Sood (23BDS0177) · Yash Poddar (23BDS0195)
**Phase:** 6 of 8
**Estimated Duration:** 2 weeks
**Depends on:** Phase 3 (compression), Phase 4 (async aggregation), Phase 5 (connectivity engine)

---

## 1. Where This Fits

Phases 3–5 each validated one dimension in isolation: compression ratio (Phase 3), dropout tolerance (Phase 4), and bandwidth realism (Phase 5). Phase 6 stops treating these as separate experiments and runs them **together, systematically**, producing the exact joint metric named in your proposed direction slide: *accuracy retained at X% bandwidth, Y% dropout*. This is also the phase where results get organized well enough to feed both the final report (Phase 8) and the dashboard (Phase 7) — sloppy, ad hoc results here become a bottleneck for both.

| Phase | Name | Status |
|---|---|---|
| 1–5 | Foundation → Connectivity Engine | ✅ done |
| **6** | **Evaluation & Experiment Orchestration** | **this file** |
| 7 | Monitoring Dashboard | depends on Phase 6 |
| 8 | Integration, Ablations & Final Report | depends on all |

---

## 2. Objectives of Phase 6

- [ ] A single orchestration mechanism can launch a full sweep (compression × dropout × bandwidth) from one config, rather than manually kicking off each run.
- [ ] Every run's full parameter set and results are written to a **persistent, queryable store** (Supabase), not just local log files — this is what Phase 7's dashboard will read from.
- [ ] The core joint metric — accuracy (and macro-AUROC/F1) retained as a function of bandwidth and dropout — is computed and visualized as a 2D surface/heatmap, not just line charts of one variable at a time.
- [ ] Results include statistical rigor: multiple seeds per configuration, with means and variance/confidence intervals reported, not single-run numbers.
- [ ] The best-performing configuration(s) — the practical "operating point" recommendation — are identified and justified with data.

---

## 3. Experiment Orchestration Design

### 3.1 Sweep dimensions

| Dimension | Values to include (minimum) | Source phase |
|---|---|---|
| Compression method + setting | None (Phase 2 baseline), top-k sparsification at 2–3 ratios, quantization at 2–3 bit-widths | Phase 3 |
| Aggregation strategy | Synchronous FedAvg, buffered async (best buffer size from Phase 4) | Phase 4 |
| Dropout/connectivity condition | At minimum: no dropout, moderate, severe (map to Phase 5's connectivity profiles rather than Phase 4's simple probabilities, since Phase 5 is the more realistic model) | Phase 5 |
| Random seed | ≥3 seeds per configuration | — |

Full factorial combination of all of the above will be large — Section 3.3 addresses how to keep this tractable.

### 3.2 Orchestration mechanism

Build a lightweight **experiment runner** that reads a single sweep-definition config (listing the grid or specific combinations to run), launches each configuration as a separate tracked run (reusing the experiment-tracking discipline from Phase 1/2), and writes results — both to the experiment tracker (W&B/MLflow) for exploration, and to Supabase (Section 5) for the dashboard. This is orchestration logic, not new ML logic — it should call the exact same client/server/aggregation/connectivity code built in Phases 2–5 unchanged, just varying which config is passed in.

### 3.3 Keeping the sweep tractable

A full factorial grid across every dimension in Section 3.1 with 3 seeds each will be expensive. Prioritize in this order if compute/time is constrained:
1. **Core joint sweep:** compression × dropout/connectivity, at the single best async buffer size from Phase 4, 3 seeds each — this alone produces the headline metric.
2. **Secondary:** sync vs. async comparison at the core sweep's most realistic dropout setting only (not swept across every dropout level again).
3. **Optional, time-permitting:** full 3-way grid including buffer size as a third axis.

Document explicitly which combinations were run and which were deliberately deprioritized — an honest "we prioritized X over Y given time constraints" is stronger for the final report than silently having gaps.

---

## 4. Statistical Rigor

- Report mean ± standard deviation (or a confidence interval) across seeds for every headline number, not single-run values — a single FedAvg run under stochastic client sampling, dropout, and quantization noise is not representative on its own.
- Where comparing two conditions (e.g., sync vs. async under the same dropout), note whether the difference is larger than the run-to-run variance before claiming one is meaningfully better.
- Keep per-seed raw results (not just the aggregate) — Phase 8's final report and any last-minute reviewer question ("show me one specific run") will need to point at a real underlying result, not just a summary statistic.

---

## 5. Persisting Results to Supabase

This is the first phase where Supabase moves from "bootstrapped shell" (Phase 1, Section 9) to an actively used store, laying the exact groundwork Phase 7's dashboard depends on.

### 5.1 Schema to implement now

| Table | Purpose | Key columns (conceptual, not exact SQL) |
|---|---|---|
| `experiments` | One row per named experiment/sweep | id, name, description, created_at, config reference |
| `runs` | One row per individual run within a sweep (one seed × one config combination) | id, experiment_id, compression_setting, aggregation_strategy, dropout_condition, seed, status |
| `rounds` | One row per training round/buffer-aggregation-step within a run | id, run_id, round_number, timestamp, global_accuracy, global_auroc, global_f1, communication_bytes |
| `clients` | One row per simulated clinic, static metadata | id, partition_id, data_volume, label_distribution_summary, connectivity_profile_id |
| `client_round_events` | One row per client's participation (or dropout/straggle) event per round | id, run_id, client_id, round_number, status (participated/dropped/late), staleness, effective_bandwidth |

This schema is intentionally normalized so the dashboard (Phase 7) can answer both "how did this run converge over time" (query `rounds`) and "which clients struggled most" (query `client_round_events`) without redundant duplicated data.

### 5.2 Writing results

Decide and document one mechanism: either the Python research code writes directly to Supabase via its REST/client API at the end of each run (simplest, real-time-capable), or results are exported to structured files (CSV/JSON) and a separate small ingestion step loads them into Supabase in batch (simpler to keep the research code free of infrastructure dependencies). **Recommendation:** batch ingestion — it keeps Phases 2–5's research code decoupled from web infrastructure, which matters if anything about the Supabase schema changes later.

---

## 6. Producing the Joint Metric Visualization

- **Primary figure:** a 2D heatmap or contour plot with bandwidth (or compression ratio, as a proxy) on one axis, dropout rate on the other, and retained accuracy (relative to the Phase 2 no-compression/no-dropout ceiling) as the color/contour value. This is the single figure that most directly answers your project's proposed direction.
- **Secondary figure:** overlay sync vs. async aggregation as two separate heatmaps (or a difference heatmap) to show where asynchronous aggregation's benefit is largest — likely at higher dropout rates, which would be a clean, citable finding.
- Identify and explicitly call out the **recommended practical operating point** — e.g., "at this compression level and this connectivity condition, you retain N% accuracy, which we judge to be the best tradeoff because ___" — grounded in the heatmap, not asserted without evidence.

---

## 7. Repository Additions for This Phase

```
research/
├── src/
│   └── eval/
│       ├── experiment_runner.py       # Orchestrates sweeps across Phases 3-5's components
│       ├── supabase_writer.py         # Batch ingestion of run/round/client results into Supabase
│       └── joint_metric_plots.py      # Heatmap/contour generation for the core joint metric
├── configs/
│   └── phase6_core_joint_sweep.yaml
infra/
└── supabase/
    └── migrations/
        └── 0001_experiments_runs_rounds_clients_schema.sql
results/
└── phase6/                             # Aggregated sweep results, heatmaps, recommended operating point writeup
```

*(Structure and responsibilities only — no code included per your instructions.)*

---

## 8. Deliverables Checklist

- [ ] Experiment runner implemented, capable of launching a full sweep from one config without manual per-run intervention.
- [ ] Supabase schema (Section 5.1) created via migration, matching what Phase 7's dashboard will need.
- [ ] Result-writing mechanism (batch ingestion, per Section 5.2) implemented and validated against at least one full run.
- [ ] Core joint sweep (Section 3.3, priority 1) completed across ≥3 seeds per configuration.
- [ ] Sync-vs-async secondary comparison (Section 3.3, priority 2) completed.
- [ ] Joint-metric heatmap produced and reviewed.
- [ ] Sync vs. async difference visualization produced, isolating where async's benefit is largest.
- [ ] Recommended operating point identified and justified with the underlying numbers.
- [ ] Mean ± variance reported for every headline number; raw per-seed results retained.
- [ ] Explicit documentation of which sweep combinations were prioritized vs. deprioritized.

---

## 9. Definition of Done

Phase 6 is complete when the joint-metric heatmap exists, is populated with real (not placeholder) data across multiple seeds, is stored queryably in Supabase, and you can point at one specific coordinate on it and say: *"here is our recommended deployment operating point, and here is why."*

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Full factorial sweep is too expensive to complete in time | Follow the prioritization order in Section 3.3 strictly; a smaller, well-documented sweep beats an incomplete, undocumented large one |
| Supabase schema needs to change after the dashboard (Phase 7) reveals new requirements | Expect at least one migration revision — design the schema to be extended, not treated as immutable after this phase |
| Single-seed anomalies get reported as if they were general findings | Enforce the mean ± variance discipline (Section 4) before any number goes into the final report |
| Heatmap looks noisy/uninterpretable due to too few seeds or too coarse a grid | Prioritize seed count and grid resolution on the core sweep over chasing every secondary comparison |

---

## 11. Next Phase Preview

**Phase 7 — Monitoring Dashboard** builds the Next.js + Express + Supabase web application that reads directly from the schema populated in this phase, giving the team (and anyone reviewing the project) a way to explore runs, compare configurations, and inspect per-client behavior without querying the database or tracker manually.
