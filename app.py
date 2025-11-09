import os
import logging
from typing import List

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

# Optional: pip install flask-cors
from flask_cors import CORS

# -----------------------------
# Config
# -----------------------------
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # In production, restrict to your GH Pages origin

MAX_CONTENT_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "10"))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILES = int(os.getenv("MAX_FILES", "12"))
MODEL_PATH = os.getenv("MODEL_PATH", "model.pth")

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

# -----------------------------
# Model
# -----------------------------
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

def get_model() -> nn.Module:
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Place it next to app.py or set MODEL_PATH."
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
    fn = secure_filename(filename or "")
    ext = os.path.splitext(fn)[1].lower()
    return ext in ALLOWED_EXTENSIONS

# -----------------------------
# Routes
# -----------------------------
@app.post("/predict")
def predict():
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files uploaded!"}), 400
    if len(files) > MAX_FILES:
        return jsonify({"error": f"Too many files. Limit is {MAX_FILES} per request."}), 400

    tensors: List[torch.Tensor] = []
    index_map: List[int] = []

    for i, f in enumerate(files):
        if not getattr(f, "filename", "") or not allowed_file(f.filename):
            continue
        try:
            img = Image.open(f.stream)
            img = ImageOps.exif_transpose(img).convert("RGB")
            tensors.append(transform(img))
            index_map.append(i)
        except (UnidentifiedImageError, OSError):
            continue

    if not tensors:
        return jsonify({"error": "No valid images. Please upload JPG/PNG/BMP/WEBP."}), 400

    batch = torch.stack(tensors, dim=0).to(device)

    model = get_model()
    with torch.inference_mode():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        real = probs[:, 0].tolist()
        fake = probs[:, 1].tolist()

    results = []
    for i, rp, fp in zip(index_map, real, fake):
        results.append({
            "index": i,  # position of the original file in the user's selection
            "label": "🔴 Fake" if fp > rp else "🟢 Real",
            "real_conf": round(rp * 100, 2),
            "fake_conf": round(fp * 100, 2),
        })

    return jsonify({"results": results}), 200

@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"Upload too large. Max {MAX_CONTENT_MB} MB per request."}), 413

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)