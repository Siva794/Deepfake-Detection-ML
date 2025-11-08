from __future__ import annotations

import os
import uuid
import logging
from typing import List, Tuple

from flask import Flask, render_template, request, abort
from werkzeug.utils import secure_filename

import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageFile, UnidentifiedImageError

# ----------------------------------
# Flask & App Config
# ----------------------------------
app = Flask(__name__)

# Max 10 MB per request (tweak as needed)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH_MB", "10")) * 1024 * 1024

# Uploads live under static so they can be served directly
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join("static", "uploads"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model location (env first, then default relative path)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "model.pth"))

# Allow only common image types
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Pillow safety: avoid truncated image errors and decompression bombs
ImageFile.LOAD_TRUNCATED_IMAGES = True
# If you expect huge images, increase this; default is to warn/raise for very big files
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "178956970"))  # ~170MP default in Pillow

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("deepfake-app")

# ----------------------------------
# Torch / Device
# ----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)


# ----------------------------------
# Model Definition (must match training)
# ----------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ----------------------------------
# Load Model
# ----------------------------------
def load_model(path: str) -> nn.Module:
    if not os.path.exists(path):
        logger.error("Model file not found at %s", path)
        raise FileNotFoundError(f"Model file not found at {path}. See README for download instructions.")
    model = SimpleCNN(num_classes=2)
    # Note: using map_location for CPU/GPU portability
    checkpoint = torch.load(path, map_location=device)
    # Handle both state_dict dump and raw state_dict
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s", path)
    return model


model = load_model(MODEL_PATH)

# ----------------------------------
# Preprocess
# ----------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ----------------------------------
# Helpers
# ----------------------------------
def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def save_image(file_storage) -> Tuple[str, Image.Image]:
    """
    Save an uploaded image to disk and return (public_path, PIL image).
    """
    original = secure_filename(file_storage.filename or "")
    if not original or not allowed_file(original):
        raise ValueError("Unsupported file type. Allowed: " + ", ".join(sorted(ALLOWED_EXTENSIONS)))

    try:
        img = Image.open(file_storage.stream).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Invalid image file.") from e

    # Generate a unique name but keep the original extension
    ext = os.path.splitext(original)[1].lower()
    filename = f"{uuid.uuid4().hex}{ext}"
    disk_path = os.path.join(UPLOAD_FOLDER, filename)
    img.save(disk_path, quality=95)

    # Public path for <img src="...">
    public_path = f"/{disk_path.replace(os.sep, '/')}"
    return public_path, img


# ----------------------------------
# Routes
# ----------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("file")
    if not files:
        return render_template("index.html", error="No files uploaded!")

    saved: List[str] = []
    tensors: List[torch.Tensor] = []

    # Collect valid images
    for f in files:
        if not getattr(f, "filename", ""):
            continue
        try:
            public_path, pil_img = save_image(f)
            saved.append(public_path)
            tensors.append(transform(pil_img))
        except ValueError as e:
            logger.warning("Skipping file: %s", e)
            continue

    if not tensors:
        return render_template("index.html", error="No valid images. Please upload JPG/PNG/BMP/WEBP.")

    batch = torch.stack(tensors, dim=0).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)  # [N, 2]
        real = probs[:, 0].tolist()
        fake = probs[:, 1].tolist()

    results = []
    for path, rp, fp in zip(saved, real, fake):
        label = "🔴 Fake" if fp > rp else "🟢 Real"
        results.append({
            "path": path,
            "label": label,
            "real_conf": round(rp * 100, 2),
            "fake_conf": round(fp * 100, 2),
        })

    return render_template("index.html", results=results)


# ----------------------------------
# Entrypoint
# ----------------------------------
if __name__ == "__main__":
    # In production, run with a WSGI server (e.g., gunicorn) and debug=False
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
