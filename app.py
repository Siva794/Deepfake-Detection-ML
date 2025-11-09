import os
import logging
from typing import List

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

# -------------------------
# App / Logging
# -------------------------
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # allow GitHub Pages to call us
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("deepfake-api")

# Safety
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "178956970"))

# -------------------------
# Torch / Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
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

MODEL_PATH = os.getenv("MODEL_PATH", "model.pth")

_model: nn.Module | None = None
def get_model() -> nn.Module:
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Place it next to app.py or set MODEL_PATH."
        )
    model = SimpleCNN(num_classes=2)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _model = model
    logger.info("Model loaded")
    return _model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

# -------------------------
# Routes
# -------------------------
@app.post("/predict")
def predict():
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    tensors: List[torch.Tensor] = []
    valid_idx = []  # keep track of which files are valid
    for idx, f in enumerate(files):
        if not getattr(f, "filename", ""):
            continue
        try:
            img = Image.open(f.stream)
            img = ImageOps.exif_transpose(img).convert("RGB")
            tensors.append(transform(img))
            valid_idx.append(idx)
        except (UnidentifiedImageError, OSError):
            # skip invalid files
            continue

    if not tensors:
        return jsonify({"error": "No valid images. Please upload JPG/PNG/BMP/WEBP."}), 400

    batch = torch.stack(tensors, dim=0).to(device)
    model = get_model()
    with torch.inference_mode():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)  # [N, 2]
        real = probs[:, 0].tolist()
        fake = probs[:, 1].tolist()

    results = []
    for rp, fp, idx in zip(real, fake, valid_idx):
        label = "Fake" if fp > rp else "Real"
        results.append({
            "index": idx,  # index in the original upload list
            "label": label,
            "real_conf": round(rp * 100, 2),
            "fake_conf": round(fp * 100, 2),
        })

    return jsonify({"results": results}), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)