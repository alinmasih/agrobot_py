import os
import io
import numpy as np
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
# Enable CORS for all routes and origins to ensure ESP32 and Flutter can connect
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# ==========================================
# 1. AI MODEL SETUP
# ==========================================
MODEL_PATH = "plant_model.tflite"
# Note: Ensure these match your Colab training labels exactly
LABELS = ["Lemon Diseased", "Lemon Healthy", "Spider Diseased", "Spider Healthy", "Background"]

try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ AI Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load AI model: {e}")

# ==========================================
# 2. GLOBAL SENSOR STATE
# ==========================================
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
    return "Agrobot AI Server is Active and Running! Use /get-sensors to see data."

@app.route('/get-sensors', methods=['GET'])
def get_sensors():
    """Hands sensor data to the WhatsApp bot or Browser."""
    return jsonify(sensor_data)

@app.route('/update-sensors', methods=['POST', 'OPTIONS'])
def update_sensors():
    """Receives data from ESP32 DevKit V1."""
    global sensor_data
    if request.method == 'OPTIONS':
        return "OK", 200
        
    try:
        # force=True ignores strict 'application/json' header requirements from ESP32
        data = request.get_json(force=True)
        if data:
            sensor_data.update(data)
            # Broadcast to Flutter App instantly via WebSockets
            socketio.emit('sensor_update', sensor_data)
            print(f"📥 Sensors updated: {data}")
            return "OK", 200
        return "No data", 400
    except Exception as e:
        print(f"❌ Update Error: {str(e)}")
        return f"Error: {str(e)}", 400

@app.route('/predict', methods=['POST'], strict_slashes=False)
def predict():
    """Receives image from ESP32-CAM and runs AI inference."""
    global sensor_data
    try:
        img_bytes = request.data
        if not img_bytes:
            return "No data received", 400

        print(f"📸 Received image: {len(img_bytes)} bytes")

        # Image Pre-processing
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
        
        # 75% Threshold to avoid fake predictions
        THRESHOLD = 0.75 
        
        if confidence >= THRESHOLD:
            prediction = LABELS[result_index]
            # Handle the 'Background' or 'No Plant' case specifically
            if "Background" in prediction:
                prediction = "No Plant Detected"
        else:
            prediction = "Searching for Plant..."

        print(f"🎯 AI Result: {prediction} ({confidence*100:.1f}%)")

        sensor_data['prediction'] = prediction
        socketio.emit('sensor_update', sensor_data)

        return prediction

    except Exception as e:
        print(f"❌ Prediction Error: {str(e)}")
        return f"Server Error: {str(e)}", 500

# ==========================================
# 4. WEBSOCKET CONTROLS
# ==========================================
@socketio.on('control_cart')
def handle_control(direction):
    """Broadcasts movement commands from Flutter/WhatsApp to ESP32."""
    print(f"🎮 Command: {direction}")
    socketio.emit('command_to_bot', direction)

# ==========================================
# 5. SERVER START
# ==========================================
if __name__ == "__main__":
    # Render assigns a dynamic port; default to 10000 for local testing
    port = int(os.environ.get("PORT", 10000))
    # Using eventlet or gevent is recommended for SocketIO in production
    socketio.run(app, host='0.0.0.0', port=port)
