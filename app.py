from flask import Flask, request, jsonify
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