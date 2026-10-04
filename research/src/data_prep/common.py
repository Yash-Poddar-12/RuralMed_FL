import hashlib
import json
from pathlib import Path
import yaml

CLASSES = ["Normal", "Pneumonia", "COVID-19"]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema")
    if abs(sum(config["split"][key] for key in ["train", "val", "test"]) - 1) > 1e-9:
        raise ValueError("Split fractions must sum to one")
    return config


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
