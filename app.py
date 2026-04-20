import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import io
from PIL import Image
import os

app = Flask(__name__)

# 1. Setup the Classifier
# Make sure 'plant_model.tflite' is in your GitHub root folder
model_path = 'plant_model.tflite'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ImageClassifierOptions(
    base_options=base_options, 
    max_results=1
)
classifier = vision.ImageClassifier.create_from_options(options)

# Labels based on your folder order
LABELS = ["Lemon Diseased", "Lemon Healthy", "Spider Diseased", "Spider Healthy"]

@app.route('/')
def home():
    return "Agrobot AI Server is Active"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive image bytes from ESP32-CAM
        img_bytes = request.data
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Convert PIL Image to MediaPipe Image format
        numpy_img = np.array(img)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_img)
        
        # 2. Run Prediction
        classification_result = classifier.classify(mp_image)
        
        # Extract the label
        # If the model has internal labels, it uses them; 
        # otherwise, we map the index to our LABELS list.
        category = classification_result.classifications[0].categories[0]
        
        # If the model didn't store names, category.index will give us the number
        result_text = category.category_name if category.category_name else LABELS[category.index]
        
        print(f"Result: {result_text}")
        return result_text

    except Exception as e:
        print(f"Error: {e}")
        return f"Server Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
