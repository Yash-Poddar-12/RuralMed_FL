"""Import every pinned component and verify CUDA build executing on CPU."""
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import albumentations
import flwr
import matplotlib
import mlflow
import numpy
import pandas
import PIL
import requests
import seaborn
import sklearn
import torch
import torchvision
import yaml

assert sys.version_info[:2] == (3, 11)
assert torch.version.cuda == "12.6"
assert torch.__version__ == "2.7.1+cu126"
assert torchvision.__version__ == "0.22.1+cu126"
image = torch.zeros(2, 3, 224, 224, device="cpu")
output = torch.nn.Conv2d(3, 4, 3)(image)
assert output.shape == (2, 4, 222, 222) and output.device.type == "cpu"
assert torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.ones(1), 0.5).tolist() == [0]
result = {"python": platform.python_version(), "executable": sys.executable,
          "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
          "cpu_tensor_operations": "passed", "torchvision_native_cpu_ops": "passed",
          "versions": {name: version(name) for name in ["torch", "torchvision", "flwr", "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "Pillow", "PyYAML", "albumentations", "mlflow", "requests"]}}
Path("research/artifacts").mkdir(exist_ok=True)
Path("research/artifacts/environment-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
