from flask import Flask, jsonify, request
import os
import pickle
import joblib
import tensorflow.lite as tflite
import numpy as np

app = Flask(__name__)

NUM_TIMESTEPS = 30
NUM_FEATURES = 11
scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler1.pkl')
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SMART_GLOVEmodel.tflite')

with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

client_buffers = {}

classes = ['B', 'C', 'D', 'GOOD MORNING', "MA'AM", 'SIR']

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

        # Clear buffer for next gesture
        sensor_buffer = []

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/echo_message', methods=['POST','GET'])
def echo_message():
    try:
        data = request.get_json()
        message = 'Nalla Vishesham!!!!'  # Extract the message from the JSON

        return jsonify({'received_message': message})  # Send it back

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run()
