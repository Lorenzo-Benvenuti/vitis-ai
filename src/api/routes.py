import logging
import random
import threading
from pathlib import Path
from typing import List

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from src.api.schemas import (
    ModelTrainRequest,
    ModelTrainResponse,
    PredictResponse,
    UploadMetadata,
    UploadResponse,
)
from src.utils.images import save_images_to_disk
from src.utils.inference import predict_variety
from src.utils.training import run_training

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

TRAINING_DIR = Path("data/training")
VALIDATION_DIR = Path("data/validation")


@router.post(
    "/model/train",
    response_model=ModelTrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def train_model(request: ModelTrainRequest):
    logger.info("POST /model/train - Starting training")

    if not TRAINING_DIR.exists() or not VALIDATION_DIR.exists():
        raise HTTPException(
            status_code=400,
            detail="Training or validation directory does not exist.",
        )

    if not any(TRAINING_DIR.iterdir()) or not any(VALIDATION_DIR.iterdir()):
        raise HTTPException(
            status_code=400,
            detail="Training and validation datasets are empty.",
        )

    # Training can run for hours, so the endpoint schedules it on a daemon thread
    # and returns instead of holding the connection open.
    thread = threading.Thread(
        target=run_training,
        kwargs={"request": request},
        daemon=True,
    )
    thread.start()

    logger.info("Training scheduled in background")

    return ModelTrainResponse(
        status="started",
        message="Training started in the background.",
    )


@router.post("/image/upload", response_model=UploadResponse)
def upload_images(
    images: List[UploadFile] = File(...),
    metadata: str = Form(...),
):
    logger.info("POST /image/upload - Starting image upload")

    try:
        parsed = UploadMetadata.model_validate_json(metadata)
    except ValidationError:
        raise HTTPException(status_code=400, detail="Invalid metadata")

    for image in images:
        extension = Path(image.filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {extension}",
            )

    # The split is per request, not global: a lone image gives int(0.8) == 0 and
    # lands entirely in validation. Upload in batches.
    random.shuffle(images)
    split_index = int(len(images) * 0.8)
    train_items = images[:split_index]
    val_items = images[split_index:]

    train_dir = TRAINING_DIR / parsed.variety
    val_dir = VALIDATION_DIR / parsed.variety

    try:
        saved_train = save_images_to_disk(train_items, train_dir)
        saved_val = save_images_to_disk(val_items, val_dir)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save images: {exc}",
        )

    logger.info(
        "Upload complete: %d training, %d validation",
        saved_train,
        saved_val,
    )

    return UploadResponse(
        uploaded=len(images),
        train_count=saved_train,
        validation_count=saved_val,
        variety=parsed.variety,
    )


@router.post("/image/predict", response_model=PredictResponse)
def predict_image(image: UploadFile = File(...)):
    logger.info("POST /image/predict - Running inference")

    # Reject bad input before loading a model into memory.
    extension = Path(image.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {extension}",
        )

    try:
        pil_image = Image.open(image.file)
        pil_image.load()  # decode now, so truncated files fail here and not mid-inference
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        variety, confidence = predict_variety(pil_image)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No trained model available. Train a model first.",
        )

    logger.info("Prediction complete: variety=%s confidence=%.4f", variety, confidence)

    return PredictResponse(variety=variety, confidence=confidence)
