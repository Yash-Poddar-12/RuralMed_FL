# Phase 8 Handoff Runbook

Phase 8 packages the existing Phase 1-7 work into an integration trace, focused
ablation plan, final report evidence, and demo checklist. It does not invent
results. Every scientific claim must point to a raw result JSON or a Supabase
row.

## 1. Generate the Phase 8 report artifacts

From the repository root:

```bash
python -m research.src.eval.phase8_report \
  --phase6-dir results/phase6/raw \
  --ablation-dir results/phase8/ablations \
  --output-dir results/phase8 \
  --batch-json results/phase6/supabase_batch.json
```

This creates:

- `results/phase8/phase8_report.md`
- `results/phase8/phase8_summary.csv`
- `results/phase8/integration_trace.json`

Missing ablation files are reported as missing evidence; they are not replaced
with placeholder metrics.

## 2. Run the focused ablations

The execution contract is [research/configs/phase8_ablations.yaml](research/configs/phase8_ablations.yaml).
It defines five focused comparisons:

1. IID versus non-IID clients
2. Error feedback enabled versus disabled
3. Staleness weighting enabled versus disabled
4. Trace-driven connectivity versus simple dropout
5. Compression-only versus compression-plus-async versus the full system

Use the existing Phase 2-5 implementations and write each completed run as a
Phase 6-compatible JSON under `results/phase8/ablations/`. Each result should
include `ablation` and `variant` fields in its configuration so it can be
identified in the report.

## 3. Complete the integration trace manually

Select one representative Phase 6 run and compare:

1. `results/phase6/raw/<run>.json`
2. The matching `runs.final_metrics` Supabase row
3. The value displayed by the Phase 7 API/dashboard

Record the result in the final presentation notes. The generated
`integration_trace.json` records the local-to-batch portion only; it cannot
verify a live Supabase or browser response without credentials and a running
dashboard.

## 4. Faculty demo path

Present the system in this order:

1. Phase 1: dataset, partition, and client contract
2. Phases 2-5: one result each showing baseline, compression, async aggregation, and connectivity behavior
3. Phase 6: joint metric heatmap and recommended operating point
4. Phase 7: dashboard overview, run detail, heatmap, and client explorer
5. Phase 8: ablation evidence, integration trace, limitations, and future work

Keep static screenshots of the dashboard, heatmaps, and the recommended
operating point as the fallback demo path.

## 5. Evidence still required from the team

- The current attached evidence matrix is in `results/phase8/evidence_matrix.md`.
- The faculty-facing summary is in `results/phase8/faculty_review_brief.md`.
- Completed Phase 8 ablation result JSONs are still optional follow-up evidence;
  do not describe them as completed unless the files exist.
- Supabase row counts and one manual raw-to-dashboard trace remain the only
  live-infrastructure item needed for the integration claim.
- Final literature-review text should be inserted into the report from the
  team's original slides or proposal.
- Faculty/demo screenshots or a deployment URL should be added before the
  presentation.