import os
import io
import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
CORS(app)
# Allow real-time connections from your Flutter App and Web Dashboard
socketio = SocketIO(app, cors_allowed_origins="*")

# ==========================================
# 1. AI MODEL SETUP
# ==========================================
MODEL_PATH = "plant_model.tflite"
LABELS = ["Lemon Diseased", "Lemon Healthy", "Spider Diseased", "Spider Healthy"]

try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("AI Model loaded successfully!")
except Exception as e:
    print(f"Failed to load AI model: {e}")

# ==========================================
# 2. GLOBAL SENSOR STATE
# ==========================================
# This holds the latest data so the Flutter app can read it anytime
sensor_data = {
    "soil": "0",
    "temp": "0",
    "hum": "0",
    "rain": "Unknown",
    "prediction": "Waiting..."
}

# ==========================================
# 3. HTTP ROUTES
# ==========================================
@app.route('/')
def home():
    try:
        # If you made the index.html Web Dashboard, this will show it
        return render_template('index.html')
    except:
        return "Agrobot AI Server is Active and Running!"

@app.route('/update-sensors', methods=['POST'])
def update_sensors():
    global sensor_data
    try:
        data = request.json
        if data:
            sensor_data.update(data)
            # Push the new sensor data to the Flutter App instantly
            socketio.emit('sensor_update', sensor_data)
            print(f"Sensors updated: {data}")
        return "OK"
    except Exception as e:
        return f"Error: {str(e)}", 400

@app.route('/predict', methods=['POST'], strict_slashes=False)
def predict():
    global sensor_data
    try:
        img_bytes = request.data
        if not img_bytes:
            return "No data received", 400

        print(f"Received image: {len(img_bytes)} bytes")

        # Pre-process Image for TFLite
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run AI Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Calculate Confidence
        result_index = np.argmax(output_data)
        confidence = output_data[result_index]
        
        # 70% Confidence Threshold to avoid random guessing
        THRESHOLD = 0.70 
        
        if confidence >= THRESHOLD:
            prediction = LABELS[result_index]
            print(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
        else:
            prediction = "No Plant Detected"
            print(f"Prediction: No Plant Detected (Low Confidence: {confidence:.2f})")

        # Update global state and push to Flutter App instantly
        sensor_data['prediction'] = prediction
        socketio.emit('sensor_update', sensor_data)

        return prediction

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return f"Server Error: {str(e)}", 500

# ==========================================
# 4. WEBSOCKET CONTROLS (FLUTTER -> ESP32)
# ==========================================
@socketio.on('control_cart')
def handle_control(direction):
    print(f"Control Command Received: {direction}")
    # Broadcast this command to the ESP32 to move the wheels
    socketio.emit('command_to_bot', direction)

# ==========================================
# 5. SERVER START
# ==========================================
if __name__ == "__main__":
    # Use Render's assigned port, or default to 10000
    port = int(os.environ.get("PORT", 10000))
    socketio.run(app, host='0.0.0.0', port=port)
