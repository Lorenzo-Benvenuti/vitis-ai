import json
import io
import threading
import pytest

from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
from src.main import app
from src.utils.model import build_model, save_model

client = TestClient(app)


# --- HELPERS ---

def _make_image_bytes(fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(100, 150, 200)).save(buf, format=fmt)
    return buf.getvalue()


def _assert_file_was_saved(directory: Path, expected_extension: str) -> Path:
    """Assert the directory holds at least one file with the given extension.

    Filenames cannot be predicted: the server renames uploads to random UUIDs.
    """
    assert directory.exists(), f"The directory {directory} was not created."
    files = list(directory.glob(f"*{expected_extension}"))
    assert len(files) > 0, f"No file with extension {expected_extension} found in {directory}"
    return files


# --- UPLOAD ---

def test_upload_single_image(tmp_path, monkeypatch):
    # chdir into a pytest temp directory so the test writes nowhere real.
    monkeypatch.chdir(tmp_path)

    img_bytes = _make_image_bytes("JPEG")
    metadata = json.dumps({"variety": "sangiovese"})

    response = client.post(
        "/api/image/upload",
        files=[("images", ("test.jpg", img_bytes, "image/jpeg"))],
        data={"metadata": metadata},
    )

    assert response.status_code == 200
    body = response.json()

    # One file means int(1 * 0.8) == 0, so the split sends it all to validation.
    assert body["uploaded"] == 1
    assert body["variety"] == "sangiovese"
    assert body["train_count"] == 0
    assert body["validation_count"] == 1

    _assert_file_was_saved(tmp_path / "data" / "validation" / "sangiovese", ".jpg")


def test_upload_multiple_images_split(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    img_bytes = _make_image_bytes("JPEG")
    files = [("images", (f"img_{i}.jpg", img_bytes, "image/jpeg")) for i in range(5)]
    metadata = json.dumps({"variety": "merlot"})

    response = client.post(
        "/api/image/upload",
        files=files,
        data={"metadata": metadata},
    )

    assert response.status_code == 200
    body = response.json()

    # 80/20 of five files is four and one.
    assert body["uploaded"] == 5
    assert body["train_count"] == 4
    assert body["validation_count"] == 1

    _assert_file_was_saved(tmp_path / "data" / "training" / "merlot", ".jpg")
    _assert_file_was_saved(tmp_path / "data" / "validation" / "merlot", ".jpg")


@pytest.mark.parametrize("filename, fmt, mime", [
    ("test.png", "PNG", "image/png"),
    ("test.jpeg", "JPEG", "image/jpeg"),
])
def test_upload_allowed_formats(tmp_path, monkeypatch, filename, fmt, mime):
    monkeypatch.chdir(tmp_path)

    img_bytes = _make_image_bytes(fmt)
    metadata = json.dumps({"variety": "sangiovese"})

    response = client.post(
        "/api/image/upload",
        files=[("images", (filename, img_bytes, mime))],
        data={"metadata": metadata},
    )

    assert response.status_code == 200
    assert response.json()["uploaded"] == 1

    # The extension survives to disk, lowercased.
    expected_ext = Path(filename).suffix.lower()
    _assert_file_was_saved(tmp_path / "data" / "validation" / "sangiovese", expected_ext)


def test_upload_to_existing_variety_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Simulates a previous session: the directory is already there.
    existing_dir = tmp_path / "data" / "validation" / "nebbiolo"
    existing_dir.mkdir(parents=True, exist_ok=True)

    img_bytes = _make_image_bytes("JPEG")
    metadata = json.dumps({"variety": "nebbiolo"})

    response = client.post(
        "/api/image/upload",
        files=[("images", ("new_photo.jpg", img_bytes, "image/jpeg"))],
        data={"metadata": metadata},
    )

    assert response.status_code == 200
    assert response.json()["uploaded"] == 1
    _assert_file_was_saved(existing_dir, ".jpg")


def test_upload_unsupported_format_returns_400(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    img_bytes = _make_image_bytes("GIF")
    metadata = json.dumps({"variety": "barbera"})

    response = client.post(
        "/api/image/upload",
        files=[("images", ("dangerous_file.gif", img_bytes, "image/gif"))],
        data={"metadata": metadata},
    )

    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_upload_invalid_metadata_returns_400(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    img_bytes = _make_image_bytes("JPEG")
    # Truncated JSON: Pydantic cannot parse it.
    invalid_metadata = '{"variety": '

    response = client.post(
        "/api/image/upload",
        files=[("images", ("test.jpg", img_bytes, "image/jpeg"))],
        data={"metadata": invalid_metadata},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid metadata"


def test_upload_invalid_variety_returns_400(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    img_bytes = _make_image_bytes("JPEG")
    # "../" would escape data/ if the variety validator let it through.
    metadata = json.dumps({"variety": "../escape"})

    response = client.post(
        "/api/image/upload",
        files=[("images", ("test.jpg", img_bytes, "image/jpeg"))],
        data={"metadata": metadata},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid metadata"


# --- TRAINING ---

def test_train_missing_directories_returns_400(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    response = client.post(
        "/api/model/train",
        json={
            "epochs": 5,
            "lr": 0.001,
            "batch_size": 8,
            "backbone": "resnet18",
            "augmentation": {},
        },
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Training or validation directory does not exist."
    )

def test_train_empty_datasets_returns_400(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    (tmp_path / "data" / "training").mkdir(parents=True)
    (tmp_path / "data" / "validation").mkdir(parents=True)

    response = client.post(
        "/api/model/train",
        json={
            "epochs": 5,
            "lr": 0.001,
            "batch_size": 8,
            "backbone": "resnet18",
            "augmentation": {},
        },
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Training and validation datasets are empty."
    )

def test_train_success_returns_202(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    training_dir = tmp_path / "data" / "training" / "sangiovese"
    validation_dir = tmp_path / "data" / "validation" / "sangiovese"

    training_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)

    # Enough to clear the "datasets are empty" check; never actually read.
    (training_dir / "image1.jpg").write_bytes(b"fake-image")
    (validation_dir / "image2.jpg").write_bytes(b"fake-image")

    # The endpoint's job is to validate and schedule, not to finish training, so
    # a stub keeps this test instant and offline. Patch the name where it is
    # used: routes.py imported it into its own namespace, and patching
    # src.utils.training would leave the endpoint bound to the real function.
    training_called = threading.Event()
    received = {}

    def _fake_run_training(request):
        received["request"] = request
        training_called.set()

    monkeypatch.setattr("src.api.routes.run_training", _fake_run_training)

    response = client.post(
        "/api/model/train",
        json={
            "epochs": 5,
            "lr": 0.001,
            "batch_size": 8,
            "backbone": "resnet18",
            "augmentation": {},
        },
    )

    assert response.status_code == 202

    body = response.json()

    assert body["status"] == "started"
    assert body["message"] == "Training started in the background."

    # The response beats the thread, so wait instead of asserting immediately.
    assert training_called.wait(timeout=5), "Training was never scheduled"

    # Parameters reached the worker intact.
    assert received["request"].epochs == 5
    assert received["request"].batch_size == 8
    assert received["request"].backbone == "resnet18"


# --- PREDICT ---

def _save_dummy_model(base_dir: Path, classes, backbone="resnet18") -> Path:
    # pretrained=False: random weights are fine here and skip the ImageNet download.
    model = build_model(num_classes=len(classes), backbone=backbone, pretrained=False)
    model_path = base_dir / "data" / "model" / f"{backbone}_20240101_000000.pth"
    save_model(model, str(model_path), classes=classes, backbone=backbone)
    return model_path


def test_predict_no_model_returns_404(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    img_bytes = _make_image_bytes("JPEG")
    response = client.post(
        "/api/image/predict",
        files=[("image", ("grape.jpg", img_bytes, "image/jpeg"))],
    )

    assert response.status_code == 404
    assert "No trained model" in response.json()["detail"]


def test_predict_unsupported_format_returns_400(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    img_bytes = _make_image_bytes("GIF")
    response = client.post(
        "/api/image/predict",
        files=[("image", ("animation.gif", img_bytes, "image/gif"))],
    )

    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_predict_invalid_image_returns_400(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Valid extension, unreadable bytes: rejected at decode, not at extension.
    response = client.post(
        "/api/image/predict",
        files=[("image", ("broken.jpg", b"this-is-not-an-image", "image/jpeg"))],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file"


def test_predict_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    classes = ["merlot", "sangiovese"]
    _save_dummy_model(tmp_path, classes)

    img_bytes = _make_image_bytes("JPEG")
    response = client.post(
        "/api/image/predict",
        files=[("image", ("grape.jpg", img_bytes, "image/jpeg"))],
    )

    assert response.status_code == 200
    body = response.json()

    # Untrained weights make the prediction arbitrary; only its shape is asserted.
    assert body["variety"] in classes
    assert 0.0 <= body["confidence"] <= 1.0
