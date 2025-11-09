import os
import io
import uuid
import logging
from pathlib import Path
from typing import List, Tuple

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

app = Flask(__name__)

MAX_CONTENT_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "10"))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILES = int(os.getenv("MAX_FILES", "12"))

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "178956970"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("deepfake-app")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)
torch.backends.cudnn.benchmark = True

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

_model: nn.Module | None = None
MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.pth")).resolve()

def get_model() -> nn.Module:
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Put model.pth next to app.py or set MODEL_PATH."
        )
    m = SimpleCNN(num_classes=2)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    m.load_state_dict(state_dict)
    m.to(device)
    m.eval()
    _model = m
    logger.info("Model loaded from %s", MODEL_PATH)
    return _model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def load_image_in_memory(file_storage) -> Tuple[Image.Image, str]:
    """Return (PIL image RGB, data_url) without writing to disk."""
    original = secure_filename(file_storage.filename or "")
    if not original or not allowed_file(original):
        raise ValueError("Unsupported file type. Allowed: " + ", ".join(sorted(ALLOWED_EXTENSIONS)))
    try:
        img = Image.open(file_storage.stream)
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Invalid image file.") from e

    # build a small-ish preview data URL (JPEG)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True)
    b64 = buf.getvalue()
    import base64
    data_url = "data:image/jpeg;base64," + base64.b64encode(b64).decode("ascii")
    return img, data_url

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Accept both GET and POST so “/predict” never 405s
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        # Redirect users who manually browse to /predict back to the form
        return render_template("index.html", error="Upload images to analyze.")

    files = request.files.getlist("file")
    if not files:
        return render_template("index.html", error="No files uploaded!")
    if len(files) > MAX_FILES:
        return render_template("index.html", error=f"Too many files. Limit is {MAX_FILES} per request.")

    previews: List[str] = []
    tensors: List[torch.Tensor] = []

    for f in files:
        if not getattr(f, "filename", ""):
            continue
        try:
            pil_img, data_url = load_image_in_memory(f)
            previews.append(data_url)                 # data URL for display
            tensors.append(transform(pil_img))        # tensor for model
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
    for data_url, rp, fp in zip(previews, real, fake):
        label = "🔴 Fake" if fp > rp else "🟢 Real"
        results.append({
            "data_url": data_url,  # embedded image
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