"""Phase 2 boundary: resolve image-ID indices and return normalized CPU tensors."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class ImagingDataset(Dataset):
    def __init__(self, processed_root, index_file, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.root = Path(processed_root)
        manifest = pd.read_csv(self.root / "primary_manifest.csv", keep_default_na=False).set_index("image_id", verify_integrity=True)
        ids = pd.read_csv(index_file).image_id
        if ids.duplicated().any() or not ids.isin(manifest.index).all():
            raise ValueError("Duplicate or unknown image ID in index")
        self.rows = manifest.loc[ids].reset_index()
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        relative = Path(row.processed_path)
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Image path escapes processed root")
        with Image.open(path) as image:
            array = np.array(image.convert("L"), dtype=np.float32, copy=True) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
        return (tensor - self.mean) / self.std, int(row.label)
