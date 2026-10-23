
"""Export Phase 6 results and ingest the normalized dashboard rows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import requests


TABLES = ("experiments", "runs", "rounds", "clients", "client_round_events")


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(results_dir.glob("*.json"))
        if path.name != "manifest.json"
    ]


def _run_key(result: dict[str, Any]) -> str:
    configuration = result.get("configuration", {})
    compression = configuration.get("compression", {}).get("name", "unknown")
    connectivity = _condition_name(configuration)
    strategy = result.get("strategy", "fedbuff")
    seed = result.get("seed", 42)
    return f"{strategy}_{compression}_{connectivity}_seed{seed}"


def _condition_name(configuration: dict[str, Any]) -> str:
    connectivity = configuration.get("connectivity", {})
    dropout_rate = connectivity.get("dropout_rate")
    names = {0.0: "no_dropout", 0.25: "moderate_dropout", 0.5: "severe_dropout", 0.75: "extreme_dropout"}
    return names.get(float(dropout_rate), str(connectivity.get("name", "unknown")))


def build_rows(
    experiment_name: str,
    results: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    experiments = [{
        "name": experiment_name,
        "description": "Phase 6 joint sweep",
        "config": {},
    }]
    runs: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    clients: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []

    for result in results:
        configuration = result.get("configuration", {})
        run_key = _run_key(result)
        compression = configuration.get("compression", {})
        connectivity = configuration.get("connectivity", {})
        runs.append({
            "external_id": run_key,
            "experiment_external_name": experiment_name,
            "compression_setting": compression.get("name", "unknown"),
            "aggregation_strategy": result.get("strategy", "fedbuff"),
            "dropout_condition": _condition_name(configuration),
            "bandwidth_kbps": configuration.get("bandwidth_kbps"),
            "seed": result.get("seed"),
            "status": "completed",
            "parameters": configuration,
            "final_metrics": result.get("final_metrics", {}),
        })

        communication = result.get("communication_rounds", [])
        for index, row in enumerate(result.get("rounds", [])):
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

        connectivity_summary = result.get("client_connectivity", {})
        for key, uptime in connectivity_summary.items():
            if not key.startswith("uptime/client_"):
                continue
            client_id = int(key.removeprefix("uptime/client_"))
            client_key = f"{experiment_name}:{client_id:02d}"
            clients[client_key] = {
                "external_id": client_key,
                "experiment_external_name": experiment_name,
                "client_index": client_id,
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
                "effective_bandwidth": connectivity_summary.get(
                    f"avg_bw_kbps/client_{client_id:02d}"
                ),
                "uptime": uptime,
            })

    run_ids = [run["external_id"] for run in runs]
    if len(run_ids) != len(set(run_ids)):
        duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
        raise ValueError(f"Duplicate Phase 6 run IDs: {duplicates}")

    round_ids = [(row["run_external_id"], row["round_number"]) for row in rounds]
    if len(round_ids) != len(set(round_ids)):
        raise ValueError("Duplicate (run_external_id, round_number) rows")

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


def _headers(key: str) -> dict[str, str]:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }


def _request_rows(
    method: str,
    url: str,
    key: str,
    table: str,
    *,
    params: dict[str, str] | None = None,
    payload: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    response = requests.request(
        method,
        f"{url.rstrip('/')}/rest/v1/{table}",
        headers=_headers(key),
        params=params,
        json=payload,
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(
            f"Supabase {method} {table} failed ({response.status_code}): "
            f"{response.text[:500]}"
        )
    return response.json() if response.content else []


def _upsert(
    url: str,
    key: str,
    table: str,
    payload: list[dict[str, Any]],
    conflict: str,
) -> None:
    if payload:
        _request_rows(
            "POST", url, key, table,
            params={"on_conflict": conflict},
            payload=payload,
        )


def _ids_by(
    url: str,
    key: str,
    table: str,
    column: str,
    values: Iterable[str],
) -> dict[str, str]:
    found: dict[str, str] = {}
    for value in values:
        rows = _request_rows(
            "GET", url, key, table,
            params={"select": f"id,{column}", column: f"eq.{value}"},
        )
        if len(rows) != 1:
            raise RuntimeError(f"Expected one {table}.{column} row for {value!r}")
        found[value] = rows[0]["id"]
    return found


def ingest_supabase(rows: dict[str, list[dict[str, Any]]], *, url: str, key: str) -> None:
    """Upsert Phase 6 rows and fill every normalized foreign-key column."""
    experiment_rows = rows.get("experiments", [])
    _upsert(url, key, "experiments", experiment_rows, "name")
    experiment_ids = _ids_by(
        url, key, "experiments", "name",
        [row["name"] for row in experiment_rows],
    )

    run_rows = [dict(row, experiment_id=experiment_ids[row["experiment_external_name"]])
                for row in rows.get("runs", [])]
    _upsert(url, key, "runs", run_rows, "external_id")
    run_ids = _ids_by(
        url, key, "runs", "external_id",
        [row["external_id"] for row in run_rows],
    )

    client_rows = [dict(row, experiment_id=experiment_ids[row["experiment_external_name"]])
                   for row in rows.get("clients", [])]
    _upsert(url, key, "clients", client_rows, "external_id")
    client_ids = _ids_by(
        url, key, "clients", "external_id",
        [row["external_id"] for row in client_rows],
    )

    round_rows = [dict(row, run_id=run_ids[row["run_external_id"]])
                  for row in rows.get("rounds", [])]
    _upsert(url, key, "rounds", round_rows, "run_external_id,round_number")

    event_rows = [
        dict(
            row,
            run_id=run_ids[row["run_external_id"]],
            client_id=client_ids[row["client_external_id"]],
        )
        for row in rows.get("client_round_events", [])
    ]
    if event_rows:
        for run_external_id in sorted({row["run_external_id"] for row in event_rows}):
            _request_rows(
                "DELETE", url, key, "client_round_events",
                params={"run_external_id": f"eq.{run_external_id}"},
            )
        _request_rows("POST", url, key, "client_round_events", payload=event_rows)


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
