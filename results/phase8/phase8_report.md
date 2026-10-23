# Phase 8 Integration and Ablation Report

## Evidence status

- Phase 6 result files discovered: `8`
- Phase 8 ablation result files discovered: `0`
- Local-to-batch trace status: `local_result_available`
- Dashboard trace: pending manual confirmation against the live API.

## Required ablations

The ablation plan is defined in `research/configs/phase8_ablations.yaml`.
Each planned control and variant must be backed by a result JSON before a scientific claim is made.

## Interpretation rules

- Report mean and standard deviation across seeds where at least three seeds exist.
- Do not claim an ablation effect when its result files are absent.
- Trace one final metric from raw JSON to Supabase and then to the dashboard before the demo.
- Keep simulation limitations and incomplete sweep coverage explicit in the final report.

## Generated artifacts

- `phase8_summary.csv` contains one row per discovered result JSON.
- `integration_trace.json` records the local result and optional batch comparison.
- `phase8_ablations.yaml` is the execution contract for the focused studies.
