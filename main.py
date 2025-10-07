from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import os, requests

# === Google Drive Model Download ===
MODEL_URL = "https://drive.google.com/uc?export=download&id=19HjW_tqFGCw-KUqdbPMeJgn1w8GmcVUZ"
MODEL_PATH = "ensemble_best.keras"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    r = requests.get(MODEL_URL)
    open(MODEL_PATH, "wb").write(r.content)
    print("✅ Model downloaded successfully!")

# === Model Load ===
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Grade 2', 'Grade 3', 'Grade 4']

print("🔁 Loading model ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

app = FastAPI(title="Nitrogen Grade Prediction API")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB").resize(IMG_SIZE)
        arr = np.expand_dims(np.array(image) / 255.0, axis=0)
        preds = model.predict(arr)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        return JSONResponse({"predicted_label": CLASS_NAMES[idx], "confidence": conf})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
