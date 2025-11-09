import os
import logging
from typing import List

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

app = Flask(__name__)

MAX_CONTENT_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "10"))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

MODEL_PATH = os.getenv("MODEL_PATH", "model.pth")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILES = int(os.getenv("MAX_FILES", "12"))

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "178956970"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("deepfake-app")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
logger.info("Using device: %s", device)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


_model = None
def get_model():
    global _model
    if _model:
        return _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = SimpleCNN()
    state = torch.load(MODEL_PATH, map_location=device)
    state = state.get("model_state_dict") if isinstance(state, dict) else state
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    _model = model
    return _model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def allowed_file(name):
    return any(name.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    files = request.files.getlist("file")

    if not files:
        return render_template("index.html", error="No files uploaded!")

    if len(files) > MAX_FILES:
        return render_template("index.html", error=f"Max {MAX_FILES} files allowed.")

    tensors = []
    names = []

    for f in files:
        if not f.filename or not allowed_file(f.filename):
            continue
        try:
            img = Image.open(f.stream)
            img = ImageOps.exif_transpose(img).convert("RGB")
            tensors.append(transform(img))
            names.append(secure_filename(f.filename))
        except (UnidentifiedImageError, OSError):
            continue

    if not tensors:
        return render_template("index.html", error="No valid images uploaded.")

    batch = torch.stack(tensors).to(device)
    model = get_model()

    with torch.inference_mode():
        logits = model(batch)
        probs = torch.softmax(logits, 1)
        real = probs[:, 0].tolist()
        fake = probs[:, 1].tolist()

    results = []
    for name, rp, fp in zip(names, real, fake):
        label = "🔴 Fake" if fp > rp else "🟢 Real"
        results.append({
            "name": name,
            "label": label,
            "real_conf": round(rp * 100, 2),
            "fake_conf": round(fp * 100, 2),
        })

    return render_template("index.html", results=results)


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", error=f"Max upload size is {MAX_CONTENT_MB} MB"), 413


if __name__ == "__main__":
    app.run("0.0.0.0", int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")