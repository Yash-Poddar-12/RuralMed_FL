"""Batch result export and optional Supabase REST ingestion for Phase 6."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import requests


TABLES = ("experiments", "runs", "rounds", "clients", "client_round_events")


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load completed raw result JSON files, excluding the manifest."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(results_dir.glob("*.json"))
        if path.name != "manifest.json"
    ]


def build_rows(experiment_name: str, results: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    experiments = [{"name": experiment_name, "description": "Phase 6 joint sweep", "config": {}}]
    runs: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    clients: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    for result in results:
        config = result.get("configuration", {})
        run_key = result["run_label"]
        runs.append({
            "external_id": run_key,
            "experiment_external_name": experiment_name,
            "compression_setting": config.get("compression", {}).get("name"),
            "aggregation_strategy": result.get("strategy"),
            "dropout_condition": config.get("connectivity", {}).get("name"),
            "bandwidth_kbps": config.get("bandwidth_kbps"),
            "seed": result.get("seed"),
            "status": "completed",
            "parameters": config,
            "final_metrics": result.get("final_metrics", {}),
        })
        for index, row in enumerate(result.get("rounds", [])):
            communication = result.get("communication_rounds", [])
            rounds.append({
                "run_external_id": run_key,
                "round_number": row["round_number"],
                "global_accuracy": row.get("test_accuracy"),
                "global_auroc": row.get("test_macro_auroc"),
                "global_f1": row.get("test_macro_f1"),
                "communication_bytes": (
                    communication[index].get("round_bytes")
                    if index < len(communication) else None
                ),
            })
        for event in result.get("client_events", []):
            client_id = int(event["client_index"])
            client_key = f"{experiment_name}:{client_id:02d}"
            events.append({
                "run_external_id": run_key,
                "client_external_id": client_key,
                "round_number": event.get("round_number"),
                "status": event["status"],
                "staleness": event.get("staleness"),
                "effective_bandwidth": event.get("effective_bandwidth"),
                "uptime": None,
            })
        connectivity = result.get("client_connectivity", {})
        for key, value in connectivity.items():
            if key.startswith("uptime/client_"):
                client_id = key.removeprefix("uptime/client_")
                client_key = f"{experiment_name}:{client_id}"
                clients[client_key] = {
                    "external_id": client_key,
                    "experiment_external_name": experiment_name,
                    "client_index": int(client_id),
                    "data_volume": None,
                    "label_distribution_summary": {},
                    "connectivity_profile_id": None,
                }
                events.append({
                    "run_external_id": run_key,
                    "client_external_id": client_key,
                    "round_number": None,
                    "status": "summary",
                    "staleness": None,
                    "effective_bandwidth": connectivity.get(
                        f"avg_bw_kbps/client_{client_id}", None
                    ),
                    "uptime": value,
                })
    return {
        "experiments": experiments,
        "runs": runs,
        "rounds": rounds,
        "clients": list(clients.values()),
        "client_round_events": events,
    }


def write_batch_json(rows: dict[str, list[dict[str, Any]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def ingest_supabase(rows: dict[str, list[dict[str, Any]]], *, url: str, key: str) -> None:
    """Insert rows in dependency order using Supabase's REST API."""
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    ids: dict[str, dict[str, str]] = {table: {} for table in TABLES}
    for table in TABLES:
        payload = rows.get(table, [])
        if not payload:
            continue
        response = requests.post(
            f"{url.rstrip('/')}/rest/v1/{table}",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(
                f"Supabase insert into {table} failed ({response.status_code}): "
                f"{response.text[:500]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export or ingest Phase 6 results")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--batch-json", type=Path, required=True)
    parser.add_argument("--ingest", action="store_true")
    args = parser.parse_args()
    results = load_results(args.results_dir)
    if not results:
        raise ValueError(f"No completed result JSON files found in {args.results_dir}")
    rows = build_rows("phase6-core-joint-sweep", results)
    write_batch_json(rows, args.batch_json)
    if args.ingest:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for --ingest")
        ingest_supabase(rows, url=url, key=key)
    print(f"Prepared {sum(len(values) for values in rows.values())} rows in {args.batch_json}")


if __name__ == "__main__":
    main()
