import os
import io
import base64
import logging
from typing import List, Tuple

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename  # still used to validate names quickly

import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

# ----------------------------------
# Flask & App Config (flat, in-memory)
# ----------------------------------
app = Flask(__name__)

MAX_CONTENT_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "10"))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

# Allowed file types (by extension)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILES = int(os.getenv("MAX_FILES", "12"))

# Pillow safety
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "178956970"))

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
torch.backends.cudnn.benchmark = True  # fixed 224x224 → OK to enable

# ----------------------------------
# Model (same architecture as training)
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

# Lazy singleton loader
_model: nn.Module | None = None
MODEL_PATH = os.getenv("MODEL_PATH", "model.pth")

def get_model() -> nn.Module:
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Place model.pth next to app.py or set MODEL_PATH."
        )
    model = SimpleCNN(num_classes=2)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device).eval()
    _model = model
    logger.info("Model loaded from %s", MODEL_PATH)
    return _model

# ----------------------------------
# Preprocess
# ----------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ----------------------------------
# Helpers (in-memory)
# ----------------------------------
def allowed_file(filename: str) -> bool:
    # Quick extension check (not security by itself; Pillow open does the heavy lifting)
    name = secure_filename(filename or "")
    ext = os.path.splitext(name)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def file_to_pil_and_data_url(file_storage) -> Tuple[Image.Image, str]:
    """
    Read upload into memory, return (PIL RGB image, base64 data URL for display).
    Nothing is written to disk.
    """
    if not getattr(file_storage, "filename", "") or not allowed_file(file_storage.filename):
        raise ValueError("Unsupported file type. Allowed: " + ", ".join(sorted(ALLOWED_EXTENSIONS)))

    # Read bytes once
    raw = file_storage.read()
    if not raw:
        raise ValueError("Empty file.")

    # Validate & normalize
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Invalid image file.") from e

    # Create small preview (to make data URL lighter); does not affect model input
    preview = img.copy()
    preview.thumbnail((512, 512))  # visual only

    buf = io.BytesIO()
    # Use PNG to avoid JPEG artifacts in preview (and works for all types)
    preview.save(buf, format="PNG", optimize=True)
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    return img, data_url

# ----------------------------------
# Routes
# ----------------------------------
@app.get("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predict():
    files = request.files.getlist("file")
    if not files:
        return render_template("index.html", error="No files uploaded!")
    if len(files) > MAX_FILES:
        return render_template("index.html", error=f"Too many files. Limit is {MAX_FILES} per request.")

    previews: List[str] = []
    tensors: List[torch.Tensor] = []

    for f in files:
        try:
            pil_img, data_url = file_to_pil_and_data_url(f)
            previews.append(data_url)
            tensors.append(transform(pil_img))
        except ValueError as e:
            logger.warning("Skipping file: %s", e)
            continue

    if not tensors:
        return render_template("index.html", error="No valid images. Please upload JPG/PNG/BMP/WEBP.")

    batch = torch.stack(tensors, dim=0).to(device)

    model = get_model()
    with torch.inference_mode():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        real = probs[:, 0].tolist()
        fake = probs[:, 1].tolist()

    results = []
    for src, rp, fp in zip(previews, real, fake):
        label = "🔴 Fake" if fp > rp else "🟢 Real"
        results.append({
            "src": src,                # base64 data URL
            "label": label,
            "real_conf": round(rp * 100, 2),
            "fake_conf": round(fp * 100, 2),
        })

    return render_template("index.html", results=results)

@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", error=f"Upload too large. Max {MAX_CONTENT_MB} MB per request."), 413

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)