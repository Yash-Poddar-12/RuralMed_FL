"""Build Phase 8 integration and ablation report artifacts from result JSONs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


METRICS = ("test_accuracy", "test_macro_auroc", "test_macro_f1")


def _canonical_run_label(result: dict[str, Any]) -> str:
    configuration = result.get("configuration", {})
    compression = configuration.get("compression", {}).get("name", "unknown")
    connectivity = configuration.get("connectivity", {})
    dropout_rate = connectivity.get("dropout_rate")
    condition_names = {
        0.0: "no_dropout",
        0.25: "moderate_dropout",
        0.5: "severe_dropout",
        0.75: "extreme_dropout",
    }
    condition = condition_names.get(float(dropout_rate), connectivity.get("name", "unknown"))
    strategy = result.get("strategy", "fedbuff")
    seed = result.get("seed", 42)
    return f"{strategy}_{compression}_{condition}_seed{seed}"


def _load_json_files(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    results = []
    for path in sorted(directory.rglob("*.json")):
        if path.name in {"manifest.json", "integration_trace.json"}:
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and "final_metrics" in value:
            results.append(value)
    return results


def _rows(results: Iterable[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        configuration = result.get("configuration", {})
        final = result.get("final_metrics", {})
        rows.append({
            "source": source,
            "run_label": _canonical_run_label(result),
            "seed": result.get("seed"),
            "strategy": result.get("strategy", configuration.get("strategy")),
            "ablation": result.get("ablation", configuration.get("ablation")),
            "variant": result.get("variant", configuration.get("variant")),
            **{metric: final.get(metric) for metric in METRICS},
        })
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["source", "run_label", "seed", "strategy", "ablation", "variant", *METRICS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _trace(phase6_results: list[dict[str, Any]], batch_path: Path | None) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "status": "local_result_available",
        "source_of_truth": "results/phase6/raw",
        "supabase_match": None,
        "dashboard_match": "pending manual trace",
    }
    if not phase6_results:
        trace["status"] = "missing_local_result"
        return trace

    result = phase6_results[0]
    trace["run_label"] = _canonical_run_label(result)
    trace["local_final_metrics"] = result.get("final_metrics", {})
    if batch_path and batch_path.exists():
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        matches = [
            row for row in batch.get("runs", [])
            if row.get("external_id") == trace["run_label"]
        ]
        trace["supabase_match"] = {
            "count": len(matches),
            "final_metrics": matches[0].get("final_metrics", {}) if matches else None,
            "metrics_equal": bool(matches and matches[0].get("final_metrics", {}) == result.get("final_metrics", {})),
        }
    return trace


def generate_report(
    phase6_dir: Path,
    ablation_dir: Path,
    output_dir: Path,
    batch_path: Path | None = None,
) -> dict[str, Any]:
    phase6 = _load_json_files(phase6_dir)
    ablations = _load_json_files(ablation_dir)
    rows = _rows(phase6, "phase6") + _rows(ablations, "phase8_ablations")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "phase8_summary.csv")
    trace = _trace(phase6, batch_path)
    (output_dir / "integration_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")

    report = [
        "# Phase 8 Integration and Ablation Report",
        "",
        "## Evidence status",
        "",
        f"- Phase 6 result files discovered: `{len(phase6)}`",
        f"- Phase 8 ablation result files discovered: `{len(ablations)}`",
        f"- Local-to-batch trace status: `{trace['status']}`",
        "- Dashboard trace: pending manual confirmation against the live API.",
        "",
        "## Required ablations",
        "",
        "The ablation plan is defined in `research/configs/phase8_ablations.yaml`.",
        "Each planned control and variant must be backed by a result JSON before a scientific claim is made.",
        "",
        "## Interpretation rules",
        "",
        "- Report mean and standard deviation across seeds where at least three seeds exist.",
        "- Do not claim an ablation effect when its result files are absent.",
        "- Trace one final metric from raw JSON to Supabase and then to the dashboard before the demo.",
        "- Keep simulation limitations and incomplete sweep coverage explicit in the final report.",
        "",
        "## Generated artifacts",
        "",
        "- `phase8_summary.csv` contains one row per discovered result JSON.",
        "- `integration_trace.json` records the local result and optional batch comparison.",
        "- `phase8_ablations.yaml` is the execution contract for the focused studies.",
    ]
    (output_dir / "phase8_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {
        "phase6_results": len(phase6),
        "ablation_results": len(ablations),
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 8 report artifacts")
    parser.add_argument("--phase6-dir", type=Path, default=Path("results/phase6/raw"))
    parser.add_argument("--ablation-dir", type=Path, default=Path("results/phase8/ablations"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase8"))
    parser.add_argument("--batch-json", type=Path, default=Path("results/phase6/supabase_batch.json"))
    args = parser.parse_args()
    print(json.dumps(generate_report(args.phase6_dir, args.ablation_dir, args.output_dir, args.batch_json), indent=2))


if __name__ == "__main__":
    main()