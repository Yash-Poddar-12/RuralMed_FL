"""Phase 1 orchestration only; no model training or federated-learning logic."""
import argparse
import json
from pathlib import Path
import mlflow
from .common import load_config, config_hash
from .preprocess import run as preprocess
from .split import run as split
from research.src.partitioning.partition import run as partition


def run(config_path):
    config = load_config(config_path)
    tracking = config["tracking"]
    mlflow.set_tracking_uri(tracking["uri"])
    if mlflow.get_experiment_by_name(tracking["experiment"]) is None:
        artifacts = Path(tracking["artifacts"]).resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(tracking["experiment"], artifact_location=artifacts.as_uri())
    mlflow.set_experiment(tracking["experiment"])
    with mlflow.start_run(run_name="phase1-" + config_hash(config)[:12]) as active:
        mlflow.set_tags({"phase": "1", "device": "cpu", "config_hash": config_hash(config)})
        mlflow.log_artifact(str(config_path), "configuration")
        mlflow.log_params({"seed": config["seed"], **config["partition"]})
        preprocess(config)
        data = split(config)
        result = partition(config)
        mlflow.log_metrics({"primary_images": len(data), "train_images": result["train_images"],
                            "centralized_test_images": result["test_images_centralized"],
                            "smallest_client": result["min_client_train"]})
        processed = Path(config["paths"]["processed"])
        partitions = Path(config["paths"]["partitions"])
        for path in [processed / "preprocessing.json", processed / "split-summary.json",
                     partitions / "partition.json", partitions / "summary.csv", partitions / "report.md",
                     partitions / "distribution.png"]:
            mlflow.log_artifact(str(path), "phase1")
        receipt = {"run_id": active.info.run_id, "experiment_id": active.info.experiment_id,
                   "tracking_uri": tracking["uri"], "config_hash": config_hash(config)}
        (partitions / "tracking.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/configs/phase1.yml")
    run(parser.parse_args().config)
