import numpy as np
from flask import Flask, request
import tflite_runtime.interpreter as tflite
import io
from PIL import Image

app = Flask(__name__)

# 1. Load the model
# Ensure plant_model.tflite is in your GitHub folder
interpreter = tflite.Interpreter(model_path="plant_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

LABELS = ["Lemon Diseased", "Lemon Healthy", "Spider Diseased", "Spider Healthy"]

@app.route('/')
def home():
    return "Agrobot Server is Active!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from ESP32-CAM
        img_bytes = request.data
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        
        # Pre-process for TFLite
        input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

        # 2. Run Prediction
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        result_index = np.argmax(output_data[0])
        
        return LABELS[result_index]
    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
