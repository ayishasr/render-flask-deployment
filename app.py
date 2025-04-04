# Flask (app.py)
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def home():
    return "<h1>Welcome to the Flask Server</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'prediction': 'This is the prediction'})

if __name__ == '__main__':
    app.run()