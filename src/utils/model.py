from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

# Transfer learning in one move: load a pre-trained backbone, then replace its
# final linear layer with a head sized to num_classes. Only the location of that
# head differs between families, hence one helper each.


def _resnet(factory, weights, num_classes, pretrained):
    m = factory(weights=weights if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _efficientnet(factory, weights, num_classes, pretrained):
    m = factory(weights=weights if pretrained else None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def _mobilenet(factory, weights, num_classes, pretrained):
    m = factory(weights=weights if pretrained else None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m


# Adding a backbone means adding one entry; SUPPORTED_BACKBONES follows.
_REGISTRY = {
    "resnet18": lambda n, p: _resnet(
        models.resnet18, models.ResNet18_Weights.DEFAULT, n, p
    ),
    "resnet50": lambda n, p: _resnet(
        models.resnet50, models.ResNet50_Weights.DEFAULT, n, p
    ),
    "efficientnet_b0": lambda n, p: _efficientnet(
        models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, n, p
    ),
    "mobilenet_v3_small": lambda n, p: _mobilenet(
        models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT, n, p
    ),
}

SUPPORTED_BACKBONES: list[str] = list(_REGISTRY.keys())


def build_model(
    num_classes: int,
    backbone: str = "resnet18",
    pretrained: bool = True,
) -> nn.Module:
    """Build a backbone whose head is adapted to num_classes.

    pretrained=False yields the bare architecture, which is what checkpoint
    loading needs: fetching ImageNet weights only to overwrite them a moment
    later would cost a needless download.
    """
    if backbone not in _REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Supported: {SUPPORTED_BACKBONES}"
        )

    return _REGISTRY[backbone](num_classes, pretrained)


def save_model(
    model: nn.Module,
    model_path: str,
    classes: list[str],
    backbone: str,
) -> None:
    """Save a self-describing checkpoint: weights, class names and backbone.

    Storing the last two makes the file sufficient on its own for inference,
    with no dependency on a dataset that may since have changed on disk.
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "backbone": backbone,
        },
        model_path,
    )


def load_checkpoint(
    model_path: str,
    device: str = "cpu",
) -> tuple[nn.Module, list[str]]:
    """Rebuild the model from a checkpoint and return it with its class list.

    The returned model is in eval mode, ready for deterministic inference.
    """
    # weights_only=True restricts unpickling to tensors and primitives, so a
    # tampered checkpoint cannot execute arbitrary code on load.
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    classes = checkpoint["classes"]

    model = build_model(
        num_classes=len(classes),
        backbone=checkpoint["backbone"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, classes
