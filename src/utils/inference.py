import logging
from pathlib import Path

import torch
from PIL import Image

from src.utils.dataset import val_transforms
from src.utils.model import load_checkpoint

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data") / "model"


def find_latest_model(model_dir: Path = MODEL_DIR) -> Path | None:
    """Return the most recently modified checkpoint, or None if there is none.

    Modification time rather than filename order, so the choice stays correct
    when backbones with different name prefixes sit side by side.
    """
    if not model_dir.exists():
        return None

    checkpoints = list(model_dir.glob("*.pth"))
    if not checkpoints:
        return None

    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def predict_variety(image: Image.Image) -> tuple[str, float]:
    """Classify a PIL image with the latest trained model.

    Returns ``(variety, confidence)``, confidence being the softmax probability
    of the predicted class. Raises FileNotFoundError if no model exists yet.
    """
    checkpoint_path = find_latest_model()
    if checkpoint_path is None:
        raise FileNotFoundError("No trained model available.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_checkpoint(str(checkpoint_path), device)

    # Reuses the validation pipeline deliberately: preprocess differently here
    # and the reported confidence would no longer mean what validation measured.
    tensor = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, index = probabilities.max(dim=1)

    variety = classes[index.item()]
    logger.info(
        "Inference | model=%s variety=%s confidence=%.4f",
        checkpoint_path.name,
        variety,
        confidence.item(),
    )
    return variety, round(confidence.item(), 4)
