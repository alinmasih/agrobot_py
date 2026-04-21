import os
import io
import numpy as np
from flask import Flask, request
from flask_cors import CORS
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
CORS(app) # Allows requests from different sources

# 1. Load the TFLite Model
# Ensure 'plant_model.tflite' is in the same folder as this file on GitHub
MODEL_PATH = "plant_model.tflite"

try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")

# Define labels based on your training order
LABELS = ["Lemon Diseased", "Lemon Healthy", "Spider Diseased", "Spider Healthy"]

@app.route('/')
def home():
    return "Agrobot AI Server is Active and Running!"

@app.route('/predict', methods=['POST'], strict_slashes=False)
def predict():
    try:
        # Get the raw image data from the ESP32-CAM POST request
        img_bytes = request.data
        
        if not img_bytes:
            print("Error: Received empty payload from ESP32")
            return "No data received", 400

        print(f"Received image data: {len(img_bytes)} bytes")

        # 2. Pre-process the image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Run Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        # Get results
        output_data = interpreter.get_tensor(output_details[0]['index'])
        result_index = np.argmax(output_data[0])
        prediction = LABELS[result_index]

        print(f"Prediction Result: {prediction}")
        return prediction

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return f"Server Error: {str(e)}", 500

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
