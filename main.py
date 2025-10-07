from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

MODEL_PATH = "ensemble_best.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Grade 2', 'Grade 3', 'Grade 4']

app = FastAPI(title="Nitrogen Grade Prediction API")

@app.on_event("startup")
async def startup_event():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(image) / 255.0, axis=0)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    return JSONResponse({"predicted_label": CLASS_NAMES[idx], "confidence": conf})
