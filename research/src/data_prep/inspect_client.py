import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from .common import CLASSES, load_config
from .dataset import ImagingDataset


def main():
    torch.set_num_threads(1)  # Small per-image checks avoid thread-launch overhead.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/configs/phase1.yml")
    parser.add_argument("--client", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    folder = Path(config["paths"]["partitions"]) / f"client_{args.client:02d}"
    dataset = ImagingDataset(config["paths"]["processed"], folder / "train.csv",
                             config["preprocessing"]["mean"], config["preprocessing"]["std"])
    # Decode every client image on CPU and check the actual model input contract.
    for i in range(len(dataset)):
        image, label = dataset[i]
        assert image.device.type == "cpu" and image.shape == (3, 224, 224)
        assert torch.isfinite(image).all() and label in range(3)
    counts = dataset.rows.label.value_counts().reindex(range(3), fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(CLASSES, counts)
    ax.set_title(f"Client {args.client:02d}: {len(dataset)} training images")
    ax.set_ylabel("Images")
    fig.savefig(folder / "label-distribution.png", dpi=150)
    plt.close(fig)
    print(f"CPU-loaded all {len(dataset)} images; counts={counts.to_dict()}; plot={folder / 'label-distribution.png'}")


if __name__ == "__main__":
    main()
