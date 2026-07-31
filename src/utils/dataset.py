from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.api.schemas import AugmentationConfig

# Layout is data/<split>/<class>/*.jpg: ImageFolder turns each subdirectory name
# into a label, so classes are never declared anywhere.
DATASET_DIR = Path("data")
TRAINING_DIR = DATASET_DIR / "training"
VALIDATION_DIR = DATASET_DIR / "validation"

# Pre-trained backbones expect inputs normalised with ImageNet statistics.
# Training and inference must apply the identical values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def convert_rgb(img):
    return img.convert("RGB")

def get_dataloaders(
    augmentation: AugmentationConfig,
    batch_size: int = 32,
    num_workers: int = 2,
):
    """Build the training and validation loaders and the class list.

    Training shuffles and augments; validation stays deterministic so its
    metrics are comparable across epochs.
    """
    train_dataset = ImageFolder(
        str(TRAINING_DIR),
        transform=build_train_transforms(augmentation),
    )
    val_dataset = ImageFolder(
        str(VALIDATION_DIR),
        transform=val_transforms,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_dataset.classes


def build_train_transforms(
    augmentation: AugmentationConfig,
) -> transforms.Compose:
    """Compose the training augmentation pipeline from the request config.

    Augmentation multiplies the effective dataset size and curbs overfitting,
    which is why it is applied to training only.
    """
    steps = [
        transforms.Lambda(convert_rgb),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    ]

    if augmentation.horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip())

    if augmentation.vertical_flip_p > 0:
        steps.append(transforms.RandomVerticalFlip(p=augmentation.vertical_flip_p))

    if augmentation.rotation > 0:
        steps.append(transforms.RandomRotation(degrees=augmentation.rotation))

    steps += [
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.ColorJitter(
            brightness=augmentation.color_jitter_brightness,
            contrast=augmentation.color_jitter_contrast,
            saturation=augmentation.color_jitter_saturation,
            hue=augmentation.color_jitter_hue,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    # RandomErasing operates on tensors, so it must follow ToTensor().
    if augmentation.random_erasing_p > 0:
        steps.append(
            transforms.RandomErasing(p=augmentation.random_erasing_p, scale=(0.02, 0.2))
        )

    return transforms.Compose(steps)


# Shared by validation and inference: the two must preprocess identically.
val_transforms = transforms.Compose([
    transforms.Lambda(convert_rgb),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
