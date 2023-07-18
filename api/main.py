from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "normal"]
@app.get("/ping")
async def ping():
    return "hello"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
    
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    
    json_data = {
        "instances": image_batch.tolist()
    }
    
    
    response = requests.post(endpoint, json=json_data)
    pred = response.json()["predictions"][0]
    index=np.argmax(pred)
    predicted_class = CLASS_NAMES[index]
    confidence = float(np.max(pred))
    return {"class":predicted_class, "confidence":confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)