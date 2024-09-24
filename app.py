import os
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Função para carregar e prever com o modelo selecionado
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_name = data['model']
    features = data['features']

    # Verificar se o modelo existe
    model_path = f'modelos/{model_name}_model.pkl'
    if not os.path.exists(model_path):
        return jsonify({'error': 'Modelo não encontrado.'}), 400
    
    # Carregar o modelo
    model = joblib.load(model_path)
    
    # Realizar a predição
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)