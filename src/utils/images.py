import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import UploadFile


def save_images_to_disk(images: List[UploadFile], target_dir: Path) -> int:
    """Save the images under UUID names and return how many were written.

    Creates target_dir and its parents if needed. Raises OSError on write
    failures, which /image/upload turns into a 500.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for image in images:
        extension = Path(image.filename).suffix.lower()
        filename = f"{uuid.uuid4()}{extension}"
        destination = target_dir / filename

        # Chunked copy: never holds a whole image in memory.
        with destination.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        saved_count += 1

    return saved_count
