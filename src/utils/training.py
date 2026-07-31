import copy
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from src.api.schemas import ModelTrainRequest
from src.utils.dataset import get_dataloaders
from src.utils.model import build_model, save_model

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data") / "model"


def run_training(request: ModelTrainRequest) -> Path:
    """Run a full training job and save the resulting weights.

    Designed to run on a background thread. Returns the path of the checkpoint
    written to ``data/model/``.
    """
    logger.info(
        "Training started | epochs=%s lr=%s batch_size=%s backbone=%s",
        request.epochs,
        request.lr,
        request.batch_size,
        request.backbone,
    )

    try:
        train_loader, val_loader, classes = get_dataloaders(
            batch_size=request.batch_size,
            augmentation=request.augmentation,
        )

        model = build_model(
            num_classes=len(classes),
            backbone=request.backbone,
        )

        # Longer runs deserve more patience before giving up on improvement.
        patience = max(5, request.epochs // 5)

        # Taken up front so the filename records when the run began, not when it ended.
        training_started_at = datetime.now()

        model = train_model(
            model,
            train_loader,
            val_loader,
            classes=classes,
            epochs=request.epochs,
            lr=request.lr,
            patience=patience,
        )

        timestamp = training_started_at.strftime("%Y%m%d_%H%M%S")
        model_path = MODEL_DIR / f"{request.backbone}_{timestamp}.pth"
        save_model(
            model,
            str(model_path),
            classes=classes,
            backbone=request.backbone,
        )

        logger.info("Training complete | model=%s classes=%s", model_path, classes)
        return model_path

    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        raise


def train_model(
    model,
    train_loader,
    val_loader,
    classes: list,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
):
    """Train the model and return it carrying the weights of its best epoch.

    The last epoch is rarely the best one, so the peak validation accuracy is
    tracked throughout and its weights restored before returning. Stops early
    after ``patience`` epochs without improvement.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, per_class = evaluate(
            model, val_loader, criterion, device, classes
        )
        scheduler.step()

        logger.info(
            "Epoch %3d/%d | train loss %.4f acc %.3f | val loss %.4f acc %.3f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
        )
        logger.info("Per-class accuracy (val): %s", per_class)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch, patience,
                )
                break

    model.load_state_dict(best_state)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch; return the mean loss and the accuracy."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, classes: list):
    """Evaluate the model without updating its weights.

    Alongside the mean loss and overall accuracy, returns a
    ``{class_name: accuracy}`` map that exposes which varieties are lagging.
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    num_classes = len(classes)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(1)

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            for label, pred in zip(labels, preds):
                class_correct[label.item()] += int(pred == label)
                class_total[label.item()] += 1

    per_class = {
        classes[i]: round(class_correct[i] / class_total[i], 4) if class_total[i] else 0.0
        for i in range(num_classes)
    }

    if total == 0:
        return 0.0, 0.0, per_class
    return total_loss / total, correct / total, per_class
