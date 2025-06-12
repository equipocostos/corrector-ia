
from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd

app = Flask(__name__)
modelo = joblib.load('modelo.pkl')
mapeo = pd.read_csv('mapeo_centros.csv')

@app.route('/')
def home():
    return 'Microservicio IA Centros de Costos'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    texto = data.get('texto', '')
    pred = modelo.predict([texto])[0]
    resultado = mapeo[mapeo['centro_de_costo'] == pred].iloc[0].to_dict()
    return jsonify({
        'centro_de_costo': resultado['centro_de_costo'],
        'codigo_centro_costo': resultado['codigo_centro_costo'],
        'ue': resultado['ue']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
