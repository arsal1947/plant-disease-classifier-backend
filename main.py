"""
FastAPI backend for the Plant Disease Classifier.

Loads the fine-tuned ResNet18 once at startup, exposes a /predict
endpoint that accepts an image upload and returns the top prediction
plus confidence.
"""

import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Class names, in the exact order the model was trained on ---
# IMPORTANT: this order must exactly match class_names from training
# (alphabetical order from ImageFolder). Do not reorder.
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = "best_resnet18_finetuned.pth"

app = FastAPI(title="Plant Disease Classifier API")

# Allow requests from your frontend (Netlify). Update the origin once
# you know your deployed frontend's URL — "*" is fine for initial testing
# but should be tightened before considering this production-ready.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model once at startup, not per-request (important for speed) ---
model = models.resnet18(weights=None)  # no need to re-download ImageNet weights
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Plant Disease Classifier API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(device)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Basic validation — reject non-image uploads early with a clear error
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image")

    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    top_prob, top_idx = torch.max(probabilities, dim=0)

    return PredictionResponse(
        predicted_class=CLASS_NAMES[top_idx.item()],
        confidence=round(top_prob.item(), 4)
    )
