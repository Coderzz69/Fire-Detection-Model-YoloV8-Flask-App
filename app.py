from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import base64
import cv2
import numpy as np

app = Flask(__name__)

# Load your trained YOLOv8 model
model = YOLO("fire_model.pt")  # Replace with your model path

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
