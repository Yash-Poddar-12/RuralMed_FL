"""Patient/duplicate grouping, stratified split, centralized test indices."""
from collections import Counter
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .common import write_json


def run(config):
    output = Path(config["paths"]["processed"])
    frame = pd.read_csv(output / "manifest.csv", keep_default_na=False)
    data = frame[frame.label >= 0].copy()
    # Union patients and exact processed pixels, also across dataset sources.
    parent = {identifier: identifier for identifier in frame.image_id}
    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier
    def union(first, second):
        parent[find(second)] = find(first)
    for key in ["patient_id", "pixel_sha256"]:
        for value, rows in frame.groupby(key, sort=True):
            if not value:
                continue
            ids = rows.image_id.tolist()
            for identifier in ids[1:]:
                union(ids[0], identifier)
    data["group_id"] = data.image_id.map(find)
    # Label disagreements on identical pixels cannot be trusted in the primary task.
    conflicts = set(data.groupby("pixel_sha256").label.nunique().loc[lambda x: x > 1].index)
    conflict_count = int(data.pixel_sha256.isin(conflicts).sum())
    data = data[~data.pixel_sha256.isin(conflicts)].copy()
    # Preserve official held-out COVID splits when provided. Duplicate groups and
    # any linked NIH patient follow the most conservative test > val > train split.
    priority = {"": -1, "train": 0, "val": 1, "test": 2}
    group_splits = {}
    if config["split"]["preserve_covid_official"]:
        strongest = data.official_split.map(priority).groupby(data.group_id).max()
        reverse = {value: key for key, value in priority.items()}
        group_splits = {group: reverse[value] for group, value in strongest.items() if value >= 0}
    # Remove redundant pixels while retaining the full provenance manifest.
    data = data.sort_values("image_id").drop_duplicates("pixel_sha256").copy()
    remaining = data[~data.group_id.isin(group_splits)]
    groups = []
    for group, rows in remaining.groupby("group_id", sort=True):
        groups.append((group, rows.dataset.mode()[0] + ":" + str(rows.label.mode()[0])))
    if groups:
        ids, strata = zip(*groups)
        def stratified(ids, strata, fraction, seed):
            counts = Counter(strata)
            usable = min(counts.values()) >= 2 and min(round(len(ids) * fraction), round(len(ids) * (1-fraction))) >= len(counts)
            left, right = train_test_split(list(ids), test_size=fraction, random_state=seed,
                                          stratify=list(strata) if usable else None)
            return left, right, usable
        train, held, first_stratified = stratified(ids, strata, 1 - config["split"]["train"], config["seed"])
        by_id = dict(groups)
        val, test, second_stratified = stratified(held, [by_id[g] for g in held],
            config["split"]["test"] / (config["split"]["val"] + config["split"]["test"]), config["seed"] + 1)
        for split, identifiers in [("train", train), ("val", val), ("test", test)]:
            group_splits.update({identifier: split for identifier in identifiers})
    else:
        first_stratified = second_stratified = None
    data["split"] = data.group_id.map(group_splits)
    if data.split.isna().any():
        raise AssertionError("Unassigned split")
    split_dir = output / "splits"
    split_dir.mkdir(exist_ok=True)
    for split in ["train", "val", "test"]:
        rows = data[data.split == split]
        if rows.empty:
            raise ValueError(f"Empty {split} split")
        rows[["image_id"]].to_csv(split_dir / f"{split}.csv", index=False)
    assert data.groupby("group_id").split.nunique().max() == 1
    assert data.groupby("pixel_sha256").split.nunique().max() == 1
    known = data[data.patient_id != ""]
    assert known.empty or known.groupby("patient_id").split.nunique().max() == 1
    data.to_csv(output / "primary_manifest.csv", index=False)
    report = {"primary_unique_images": len(data), "conflicting_duplicate_rows_excluded": conflict_count,
              "duplicates_removed": int((frame.label >= 0).sum()) - len(data) - conflict_count,
              "group_stratification": [first_stratified, second_stratified],
              "counts": data.groupby(["dataset", "split", "label_name"]).size().reset_index(name="images").to_dict("records"),
              "patient_leakage": 0, "exact_processed_pixel_leakage": 0,
              "limitation": "COVID-QU-Ex has no patient IDs; preserve supplied splits, but unseen patient overlap and near-duplicates cannot be ruled out."}
    write_json(output / "split-summary.json", report)
    return data


if __name__ == "__main__":
    import argparse
    from .common import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/configs/phase1.yml")
    run(load_config(parser.parse_args().config))
