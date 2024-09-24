import os
import joblib
import pytest
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Definir o threshold de desempenho (50%)
THRESHOLD = 0.50

# Função para carregar os dados e preprocessar
def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=columns)

    # Verificar se há valores faltantes
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Separar dados entre features (X) e target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar os dados
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Carregar os dados
X_train_scaled, X_test_scaled, y_train, y_test = load_data()

# Função para calcular métricas de desempenho
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, f1

# Lista dos modelos a serem testados (otimizados e não otimizados)
MODELS = [
    "../modelos/knn_model.pkl",
    "../modelos/tree_model.pkl",
    "../modelos/nb_model.pkl",
    "../modelos/svm_model.pkl",
    "../modelos/knn_model_otimizado.pkl",
    "../modelos/tree_model_otimizado.pkl"
]

# Teste automatizado para garantir que o modelo atenda ao threshold mínimo
@pytest.mark.parametrize("model_path", MODELS)
def test_model_performance(model_path):
    # Carregar o modelo
    if not os.path.exists(model_path):
        pytest.fail(f"O modelo {model_path} não foi encontrado.")

    model = joblib.load(model_path)

    # Avaliar o modelo
    accuracy, f1 = evaluate_model(model, X_test_scaled, y_test)

    # Exibir os resultados
    print(f"\nTestando o modelo: {model_path}")
    print(f"Acurácia: {accuracy:.4f}, F1-Score: {f1:.4f}")

    # Verificar se o modelo atende ao threshold
    assert accuracy >= THRESHOLD, f"Modelo {model_path} falhou no teste de acurácia! Acurácia: {accuracy:.4f}"
    assert f1 >= THRESHOLD, f"Modelo {model_path} falhou no teste de F1-Score! F1-Score: {f1:.4f}"
