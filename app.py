from flask import Flask, jsonify, request
import redis
import os
import joblib
import tensorflow.lite as tflite
import numpy as np

app = Flask(__name__)

redis_host = os.environ.get('REDIS_HOST')
redis_port = os.environ.get('REDIS_PORT')
redis_password = os.environ.get('REDIS_PASSWORD')

NUM_TIMESTEPS = 30
NUM_FEATURES = 11
scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/scaler.joblib')
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/SMART_GLOVEmodel.tflite')

try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None 

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

client_buffers = {}

classes = ['B', 'C', 'D', 'GOOD MORNING', "MA'AM", 'SIR']

if redis_host and redis_port:
    redis_client = redis.Redis(
        host=redis_host,
        port=int(redis_port),
        password=redis_password,
        decode_responses=True # Important for string handling
    )
else:
    print("Redis environment variables not set. Using localhost for development.")
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True) #fallback for local testing.

@app.route('/', methods=['GET'])
def home():
    return "<h1>Welcome to the Flask Server</h1>"

@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        data = request.get_json()  # Get JSON data from the request
        sensor_buffer = data['sensor_values']

        print("Obtained 30 timesteps, making prediction...")
        print(sensor_buffer)

        # Convert to NumPy array and apply StandardScaler
        input_data = np.array(sensor_buffer, dtype=np.float32)
        input_data = (input_data - scaler.mean_) / scaler.scale_

        # Reshape for model input
        input_data = input_data.reshape(1, NUM_TIMESTEPS, NUM_FEATURES)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get predicted class
        predicted_class = np.argmax(output_data)
        gesture = classes[predicted_class]

        print(f"Predicted Gesture: {gesture}")

        redis_client.set('gesture', gesture) #save to redis.

        # Clear buffer for next gesture
        sensor_buffer = []

        return jsonify({'prediction': gesture})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/echo_message', methods=['POST','GET'])
def echo_message():
    try:
        gesture = redis_client.get('gesture')
        if gesture:
            gesture = gesture.decode('utf-8')
        else:
            gesture = 'No gesture found'
        return jsonify({'received_message': gesture})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run()
