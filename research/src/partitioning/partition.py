"""Source skew x within-source Dirichlet label skew x long-tailed volumes."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from research.src.data_prep.common import CLASSES, config_hash, load_config, write_json


def allocate(data, rng, volumes, source_weights, label_weights, sources, minimum):
    n = len(volumes)
    budgets = volumes * len(data)
    allocated = [[] for _ in range(n)]
    sizes = np.zeros(n, dtype=int)
    groups = list(data.groupby("group_id", sort=True))
    rng.shuffle(groups)
    # Large patient groups first to avoid one group dominating a nearly full client.
    groups.sort(key=lambda entry: len(entry[1]), reverse=True)
    for _, rows in groups:
        preference = np.zeros(n)
        if len(rows) == 1:
            row = rows.iloc[0]
            counts = {(row.dataset, row.label): 1}
        else:
            counts = rows.groupby(["dataset", "label"]).size()
        for (source, label), count in counts.items():
            source_index = sources.index(source)
            preference += count * source_weights[:, source_index] * label_weights[:, source_index, label]
        capacity = np.maximum(budgets - sizes, 0)
        weights = preference * capacity
        if weights.sum() == 0:
            weights = preference / (sizes + 1)
        weights /= weights.sum()
        client = int(rng.choice(n, p=weights))
        allocated[client].append(rows)
        sizes[client] += len(rows)
    # Repair only if required, moving whole patient/duplicate groups.
    for client in range(n):
        while sizes[client] < minimum:
            donor = int(np.argmax(sizes))
            options = [(len(g), i) for i, g in enumerate(allocated[donor]) if sizes[donor] - len(g) >= minimum]
            if not options:
                raise ValueError("Cannot meet minimum client volume with indivisible groups")
            _, index = min(options)
            moved = allocated[donor].pop(index)
            allocated[client].append(moved)
            sizes[donor] -= len(moved)
            sizes[client] += len(moved)
    return [pd.concat(groups) if groups else data.iloc[:0] for groups in allocated]


def run(config):
    root = Path(config["paths"]["partitions"])
    root.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(Path(config["paths"]["processed"]) / "primary_manifest.csv", keep_default_na=False)
    params = config["partition"]
    n = params["clients"]
    if not 8 <= n <= 15 or params["alpha"] <= 0 or not 0 < params["source_preference"] < 1:
        raise ValueError("Invalid partition parameters")
    rng = np.random.default_rng(config["seed"])
    sources = sorted(data.dataset.unique())
    volumes = rng.lognormal(0, params["volume_sigma"], n)
    volumes /= volumes.sum()
    preference = params["source_preference"]
    source_weights = np.ones((n, len(sources)))
    if len(sources) > 1:
        source_weights[:] = (1 - preference) / (len(sources) - 1)
        for client in range(n):
            source_weights[client, client % len(sources)] = preference
    label_weights = rng.dirichlet([params["alpha"]] * 3, size=(n, len(sources)))
    rows, assignments = [], []
    for split in ["train", "val"]:
        subset = data.loc[data.split == split, ["image_id", "group_id", "dataset", "label", "label_name", "split"]]
        minimum = params["min_train_images"] if split == "train" else 1
        if len(subset) < n * minimum:
            raise ValueError(f"Insufficient {split} samples for {n} clients")
        partitions = allocate(subset, rng, volumes, source_weights, label_weights, sources, minimum)
        for client, part in enumerate(partitions):
            name = f"client_{client:02d}"
            folder = root / name
            folder.mkdir(exist_ok=True)
            part[["image_id"]].sort_values("image_id").to_csv(folder / f"{split}.csv", index=False)
            assignment = part[["image_id", "group_id", "dataset", "label", "label_name", "split"]].copy()
            assignment["client"] = name
            assignments.append(assignment)
            summary = {"client": name, "split": split, "images": len(part),
                       **{label: int((part.label == i).sum()) for i, label in enumerate(CLASSES)},
                       **{f"source_{source}": int((part.dataset == source).sum()) for source in sources}}
            rows.append(summary)
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "summary.csv", index=False)
    assignment = pd.concat(assignments)
    assert not assignment.image_id.duplicated().any()
    assert set(assignment.image_id) == set(data[data.split.isin(["train", "val"])].image_id)
    assert assignment.groupby("group_id").client.nunique().max() == 1
    assert not set(assignment.image_id) & set(data[data.split == "test"].image_id)
    assignment.to_csv(root / "assignments.csv", index=False)
    train = summary[summary.split == "train"].set_index("client")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    train[CLASSES].plot.bar(stacked=True, ax=axes[0], title="Training class counts")
    train[CLASSES].div(train.images, axis=0).plot.bar(stacked=True, ax=axes[1], title="Training class fractions")
    source_columns = [f"source_{source}" for source in sources]
    train[source_columns].div(train.images, axis=0).plot.bar(stacked=True, ax=axes[2], title="Dataset source fractions")
    for ax in axes:
        ax.set_xlabel("Clinic")
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Images")
    axes[1].set_ylabel("Fraction")
    axes[2].set_ylabel("Fraction")
    fig.savefig(root / "distribution.png", dpi=150)
    plt.close(fig)
    validation = {"schema_version": 1, "config_hash": config_hash(config),
                  "parameters": params, "sources": sources, "clients": n,
                  "train_images": int(train.images.sum()), "val_images": int(summary[summary.split == "val"].images.sum()),
                  "test_images_centralized": int((data.split == "test").sum()),
                  "min_client_train": int(train.images.min()), "max_client_train": int(train.images.max()),
                  "volume_ratio": float(train.images.max() / train.images.min()),
                  "source_primary": [sources[c % len(sources)] for c in range(n)],
                  "zero_client_overlap": True, "complete_coverage": True, "patient_groups_intact": True}
    write_json(root / "partition.json", validation)
    table = summary.to_string(index=False)
    (root / "report.md").write_text("# Phase 1 partition report\n\nConfig hash: " + config_hash(config) +
        "\n\nSource preferences, within-source Dirichlet draws and shared lognormal volume targets determine whole-group allocation. "
        "Train and validation are assigned separately; test stays centralized. Minimum-volume repair may weaken skew for the smallest clients.\n\n```text\n" + table +
        "\n```\n\n![Class and source distributions](distribution.png)\n", encoding="utf-8")
    print(table, flush=True)
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/configs/phase1.yml")
    run(load_config(parser.parse_args().config))
