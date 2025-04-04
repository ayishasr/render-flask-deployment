from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "<h1>Welcome to the Flask Server</h1>"

@app.route('/predict', methods=['GET'])
def predict():
    return "<h1>Prediction Endpoint</h1>"

if __name__ == '__main__':
    app.run()