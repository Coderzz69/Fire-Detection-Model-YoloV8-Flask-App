from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import base64
import cv2
import numpy as np
import requests
import os

app = Flask(__name__)

# Replace this with your actual SAS URL
MODEL_URL = "https://firedetstorageaccc.blob.core.windows.net/models/fire_model.pt?sp=r&st=2025-08-01T16:27:05Z&se=2025-08-02T00:42:05Z&spr=https&sv=2024-11-04&sr=b&sig=vZ3WLLNbd0KsJU6ystZXasTlkY7S3t5WSEv%2B0fCgwSo%3D"
MODEL_LOCAL_PATH = "fire_model.pt"

def download_model_if_needed():
    if not os.path.exists(MODEL_LOCAL_PATH):
        print("Downloading model from Azure Blob...")
        response = requests.get(MODEL_URL)
        with open(MODEL_LOCAL_PATH, "wb") as f:
            f.write(response.content)

download_model_if_needed()
model = YOLO(MODEL_LOCAL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")
    image_np = np.array(image)

    # Run detection
    results = model(image_np)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        class_name = model.names[cls_id]
        detections.append({
            "class": class_name,
            "confidence": round(conf, 2)
        })

    # Annotate image
    annotated_img = results.plot()
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "detections": detections,
        "annotated_image_base64": img_base64
    })

if __name__ == "__main__":
    app.run(debug=True)

application = app
