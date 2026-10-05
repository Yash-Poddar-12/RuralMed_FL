import hashlib
import json
import os
from pathlib import Path
import yaml

CLASSES = ["Normal", "Pneumonia", "COVID-19"]
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_config(path, profile=None):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema")
    if abs(sum(config["split"][key] for key in ["train", "val", "test"]) - 1) > 1e-9:
        raise ValueError("Split fractions must sum to one")
    dotenv = REPO_ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("RURALMED_DATA_ROOT="):
                os.environ.setdefault("RURALMED_DATA_ROOT", line.split("=", 1)[1].strip().strip("\"'"))
    data = config.setdefault("data", {"profile": "full"})
    if profile:
        data["profile"] = profile
    if data["profile"] not in {"dev", "full"}:
        raise ValueError("data.profile must be dev or full")
    if data["profile"] == "dev":
        base = REPO_ROOT / "research/data/dev_sample"
        config["paths"] = {"raw": str(base / "raw"), "processed": str(base / "processed"),
                           "partitions": str(base / "partitions")}
        config["partition"]["min_train_images"] = 5
    else:
        base = Path(os.environ.get("RURALMED_DATA_ROOT") or REPO_ROOT / "research/data").expanduser().resolve()
        for key in ("raw", "processed"):
            original = Path(config["paths"][key]).relative_to("research/data")
            config["paths"][key] = str(base / original)
        config["paths"]["partitions"] = str(REPO_ROOT / config["paths"]["partitions"])
        if data.get("cohort") == "auto":
            settings = config["datasets"]["chexpert"]
            candidates = [base / "raw/chexpert", base / "raw/chexpert/CheXpert-v1.0-small",
                          base / "raw/chexpert/extracted/CheXpert-v1.0-small"]
            chex = next((p for p in candidates if (p / settings["csv"]).is_file()), None)
            settings["enabled"] = chex is not None
            if chex:
                settings["directory"] = chex.relative_to(base / "raw").as_posix()
            data["cohort"] = "all" if chex else "public"
            for key in ("processed", "partitions"):
                config["paths"][key] = str(Path(config["paths"][key]) / data["cohort"])
    tracking = config.get("tracking", {})
    if tracking.get("uri", "").startswith("file:"):
        tracking["uri"] = (REPO_ROOT / tracking["uri"][5:]).resolve().as_uri()
    tracking["artifacts"] = str(REPO_ROOT / tracking["artifacts"])
    return config


def config_hash(config):
    portable = {key: value for key, value in config.items() if key not in {"paths", "tracking"}}
    return hashlib.sha256(json.dumps(portable, sort_keys=True).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)
