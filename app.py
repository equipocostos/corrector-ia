
from flask import Flask, request, jsonify
import os
import joblib

app = Flask(__name__)
modelo = joblib.load('modelo.pkl')

@app.route('/')
def home():
    return 'Microservicio de corrección de oficinas'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    texto = data.get('texto', '')
    pred = modelo.predict([texto])[0]
    return jsonify({'corregido': pred})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
