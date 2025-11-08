from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os
import uuid
from torch import nn

# ======================
# Flask Setup
# ======================
app = Flask(__name__)

MODEL_PATH = r"C:\Users\ksiva\Desktop\Deepfake-Detection\vit-deepfake-model\model.pth"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Model Definition (same as training)
# ======================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ======================
# Load Model
# ======================
model = SimpleCNN(num_classes=2)
checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()


# ======================
# Transform
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ======================
# Routes
# ======================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("file")
    if not files or files[0].filename == "":
        return render_template("index.html", error="No files uploaded!")

    results = []

    for file in files:
        try:
            img = Image.open(file).convert("RGB")
        except Exception:
            continue  # skip invalid files

        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        img.save(save_path)

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            real_prob = probs[0].item()
            fake_prob = probs[1].item()

        if fake_prob > real_prob:
            label = "🔴 Fake"
        else:
            label = "🟢 Real"

        results.append({
            "path": f"/{save_path}",
            "label": label,
            "real_conf": round(real_prob * 100, 2),
            "fake_conf": round(fake_prob * 100, 2)
        })

    return render_template("index.html", results=results)


# ======================
# Run App
# ======================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
