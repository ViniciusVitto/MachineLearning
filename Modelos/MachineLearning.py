import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Passo 1: Carregar o dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, names=columns)

# Verificar se há valores faltantes
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Dividir entre features (X) e target (y)
X = df.drop('target', axis=1)
y = df['target']

# Separar dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 2: Normalização
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar e salvar todos os modelos na pasta 'modelos'
if not os.path.exists('modelos'):
    os.makedirs('modelos')

# Função para calcular e exibir as métricas
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\nDesempenho do {name} antes da otimização:")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return accuracy, f1

# Função para treinar e salvar o modelo
def train_and_save_model(model, name):
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f'modelos/{name}_model.pkl')
    return model

# Treinar KNN
knn = KNeighborsClassifier()
train_and_save_model(knn, 'knn')
evaluate_model(knn, X_test_scaled, y_test, 'KNN')

# Treinar Árvore de Decisão
tree = DecisionTreeClassifier()
train_and_save_model(tree, 'tree')
evaluate_model(tree, X_test_scaled, y_test, 'Árvore de Decisão')

# Treinar Naive Bayes
nb = GaussianNB()
train_and_save_model(nb, 'nb')
evaluate_model(nb, X_test_scaled, y_test, 'Naive Bayes')

# Treinar SVM com GridSearch
param_grid_svm = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
grid_svm = GridSearchCV(SVC(), param_grid_svm, refit=True, cv=5)
grid_svm.fit(X_train_scaled, y_train)
joblib.dump(grid_svm, 'modelos/svm_model.pkl')
evaluate_model(grid_svm, X_test_scaled, y_test, 'SVM')

print("\nTodos os modelos foram treinados e salvos na pasta 'modelos'.")

# Função para otimizar e salvar o modelo otimizado
def optimize_and_save_model(model, param_grid, name):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_
    joblib.dump(best_model, f'modelos/{name}_model_otimizado.pkl')
    print(f"\nMelhor hiperparâmetro para {name}: {grid.best_params_}")
    return best_model

# Otimização de KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
best_knn = optimize_and_save_model(KNeighborsClassifier(), param_grid_knn, 'knn')
evaluate_model(best_knn, X_test_scaled, y_test, 'KNN Otimizado')

# Otimização de Árvore de Decisão
param_grid_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
best_tree = optimize_and_save_model(DecisionTreeClassifier(), param_grid_tree, 'tree')
evaluate_model(best_tree, X_test_scaled, y_test, 'Árvore de Decisão Otimizada')

# SVM já otimizado anteriormente com GridSearch
best_svm = grid_svm.best_estimator_
evaluate_model(best_svm, X_test_scaled, y_test, 'SVM Otimizado')


print("\nComparação de desempenho final realizada com sucesso!")
