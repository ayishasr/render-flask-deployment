from flask import Flask, jsonify, request
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "<h1>Welcome to the Flask Server</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get JSON data from the request
        message = data.get('message')  # Extract the 'message' from the JSON

        if message:
            print(f"Received message: {message}")  # Print the message to the server's console

            # Your prediction logic here (replace with your actual code)
            prediction = f"Prediction for '{message}' is: Some Result"

            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'Message not found in request'}), 400 #return error if message is not found.

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
    