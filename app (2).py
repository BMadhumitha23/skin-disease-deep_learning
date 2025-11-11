import os
from flask import Flask, request, render_template
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import pickle
import numpy as np
import cv2

# ======================
# 1. Flask App Setup
# ======================
app = Flask(__name__, template_folder="templates", static_folder="static")

device = torch.device("cpu")
print(f"🧠 Using device: {device}")

# ======================
# 2. Load Label Encoders
# ======================
with open("disease_encoder.pkl", "rb") as f:
    disease_encoder = pickle.load(f)
with open("severity_encoder.pkl", "rb") as f:
    severity_encoder = pickle.load(f)

# ======================
# 3. Define Model
# ======================
class MultiTaskViTModel(nn.Module):
    def __init__(self, num_diseases, num_severities):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)
        self.vit.heads = nn.Identity()
        self.disease_head = nn.Linear(768, num_diseases)
        self.severity_head = nn.Linear(768, num_severities)
        self.activations = None
        # Register hook to capture last-layer activations
        self.vit.encoder.layers[-1].register_forward_hook(self.save_activations)

    def save_activations(self, module, input, output):
        self.activations = output[:, 1:, :]  # Exclude CLS token

    def forward(self, x):
        features = self.vit(x)
        disease_out = self.disease_head(features)
        severity_out = self.severity_head(features)
        return disease_out, severity_out


# ======================
# 4. Load Model
# ======================
num_diseases = len(disease_encoder.classes_)
num_severities = len(severity_encoder.classes_)

model = MultiTaskViTModel(num_diseases, num_severities).to(device)
model.load_state_dict(torch.load("new_best_vit_model.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully!")

# ======================
# 5. Image Transform
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

# ======================
# 6. Grad-CAM Generator
# ======================
def generate_gradcam(image_pil, model, predicted_label):
    try:
        activations = model.activations
        if activations is None:
            print("⚠️ No activations found for Grad-CAM.")
            return None

        # Compute Grad-CAM
        cam = activations.mean(-1).detach().cpu().numpy()[0]
        cam = cam.reshape(14, 14)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = cv2.resize(cam, (224, 224))
        cam = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend with original image
        img_np = np.array(image_pil.resize((224, 224)))
        overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

        gradcam_filename = f"gradcam_{predicted_label.lower().replace(' ', '_')}.jpg"
        gradcam_path = os.path.join("static", gradcam_filename)
        Image.fromarray(overlay).save(gradcam_path)
        print(f"✅ Grad-CAM saved: {gradcam_path}")
        return gradcam_filename
    except Exception as e:
        print("❌ Grad-CAM generation failed:", e)
        return None


# ======================
# 7. Routes
# ======================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No file selected", 400

    os.makedirs("static", exist_ok=True)
    image_path = os.path.join("static", file.filename)
    file.save(image_path)
    image_pil = Image.open(image_path).convert("RGB")

    # Transform
    img_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        disease_out, severity_out = model(img_tensor)
        disease_probs = F.softmax(disease_out, dim=1).cpu().numpy()[0]
        severity_probs = F.softmax(severity_out, dim=1).cpu().numpy()[0]

    # Predictions
    disease_idx = np.argmax(disease_probs)
    severity_idx = np.argmax(severity_probs)
    disease_label = disease_encoder.inverse_transform([disease_idx])[0]
    severity_label = severity_encoder.inverse_transform([severity_idx])[0]

    # Optional confidence print for debugging
    print("\n📊 Disease Probabilities:")
    for i, label in enumerate(disease_encoder.classes_):
        print(f"{label}: {disease_probs[i]:.3f}")

    print("\n📊 Severity Probabilities:")
    for i, label in enumerate(severity_encoder.classes_):
        print(f"{label}: {severity_probs[i]:.3f}")

    # Confidence-based normal filter (optional)
    max_prob = np.max(disease_probs)
    if max_prob < 0.7:
        disease_label = "Normal Skin"
        severity_label = "None"

    # Grad-CAM generation
    gradcam_filename = generate_gradcam(image_pil, model, disease_label)

    return render_template(
        "result.html",
        disease=disease_label,
        severity=severity_label,
        original_image=file.filename,
        gradcam_image=gradcam_filename
    )


# ======================
# 8. Run App
# ======================
if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=True)
