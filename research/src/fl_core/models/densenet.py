"""DenseNet-121 wrapper for 3-class chest X-ray classification.

Architecture choice (for the methods section):
    DenseNet-121 (Huang et al., 2017) is the standard backbone for chest
    radiograph classification following CheXNet (Rajpurkar et al., 2017) and
    the CheXpert benchmark (Irvin et al., 2019).  Dense connections re-use
    feature maps across layers, which improves gradient flow under the small
    effective batch sizes inherent to federated training.

    The original 1000-class ImageNet head is replaced by:
        Dropout(p=0.2) → Linear(1024, 3)
    Full end-to-end fine-tuning is used (no frozen layers) because medical
    imaging domain shift from ImageNet is substantial.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights


def build_densenet121(
    num_classes: int = 3,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Return a DenseNet-121 ready for federated fine-tuning.

    Args:
        num_classes: Output classes (3 for Normal/Pneumonia/COVID-19).
        pretrained:  Load ImageNet-1k weights before replacement of head.
        dropout:     Dropout probability before the final linear layer.

    Returns:
        model: ``nn.Module`` with all parameters on CPU; caller moves to device.
    """
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)

    # Replace classifier: original is Linear(1024, 1000)
    in_features: int = model.classifier.in_features  # 1024
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_parameters(model: nn.Module) -> list:
    """Extract all trainable parameters and buffers as a list of numpy arrays.

    Flower's NumPyClient interface expects this exact format. We use state_dict
    to ensure BatchNorm running statistics are federated.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list) -> None:
    """Load a list of numpy arrays back into the model's parameters in-place."""
    import numpy as np
    from collections import OrderedDict
    
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(np.array(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
