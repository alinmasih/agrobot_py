import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
from PIL import Image

app = Flask(__name__)

# 1. Load the model you trained
model = load_model('plant_model.h5')

# 2. Define your labels (Check your Colab output for the exact order!)
# Usually: 0: lemon_diseased, 1: lemon_healthy, 2: spider_diseased, 3: spider_healthy
LABELS = ["Lemon Diseased", "Lemon Healthy", "Spider Diseased", "Spider Healthy"]

@app.route('/')
def home():
    return "Agrobot Server is Live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from the ESP32-CAM request
        img_bytes = request.data
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))
        
        # Pre-process for AI
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make Prediction
        predictions = model.predict(img_array)
        result_index = np.argmax(predictions)
        prediction_text = LABELS[result_index]

        return prediction_text # Send just the text back to the Robot
    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)