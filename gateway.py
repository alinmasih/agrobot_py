import requests
from flask import Flask, request

app = Flask(__name__)
# Your Render Cloud URL
RENDER_URL = "https://agrobot-server.onrender.com/update-sensors"

@app.route('/relay', methods=['POST'])
def relay():
    try:
        data = request.json
        print(f"\n🚜 [LOCAL] Received from ESP32: {data}")
        
        # Instantly forward the exact data to Render
        response = requests.post(RENDER_URL, json=data)
        print(f"☁️ [CLOUD] Forwarded to Render! Status: {response.status_code}")
        
        return "OK", 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return "Error", 500

if __name__ == '__main__':
    print("🌐 Agrobot Edge Gateway Running...")
    print("Waiting for ESP32 data on Port 8080...")
    # 0.0.0.0 allows the ESP32 to see the laptop on the WiFi network
    app.run(host='0.0.0.0', port=8080)
