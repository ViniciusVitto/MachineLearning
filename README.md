# MachineLearning

# Previsão de Doenças Cardíacas com Machine Learning

Este projeto aplica técnicas de machine learning para prever a presença de doenças cardíacas em pacientes, utilizando o **Heart Disease Dataset** da UCI. Através da comparação de diferentes algoritmos de classificação, buscamos otimizar os modelos e selecionar aquele com o melhor desempenho.

## Objetivos

1. Treinar diferentes modelos de machine learning (KNN, Árvore de Decisão, Naive Bayes, SVM) para prever doenças cardíacas.
2. Otimizar os hiperparâmetros dos modelos (KNN, Árvore de Decisão e SVM) utilizando `GridSearchCV`.
3. Avaliar os modelos com base nas métricas de **acurácia** e **F1-Score**.
4. Implementar testes automatizados usando **PyTest** para garantir que os modelos atendam aos requisitos mínimos de desempenho (50% de acurácia).

## Estrutura do Projeto

- **modelos**: Pasta onde os modelos treinados e otimizados são salvos.
- **notebook.ipynb**: Notebook do Google Colab com o código do projeto, incluindo a explicação do problema.
- **test_model.py**: Script de teste automatizado usando PyTest.
- **templates**: Pasta onde o Front-End está localizado.
- **README.md**: Este arquivo de documentação.

## Requisitos

- Python 3.x
- Bibliotecas necessárias:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `joblib`
  - `pytest`

Para instalar todas as dependências, execute:

```bash
pip install flask pytest
```

## Como executar

- Vá na pasta **modelos** e execute o código **MachineLearning.py**
![image](https://github.com/user-attachments/assets/c5d6481c-5bb4-4bc5-8c60-16f0b315e7d0)
- Após isso, basta executar o comando abaixo no terminal e acessar o IP **127.0.0.1:5000**:
```bash
python app.py
```
- Para executar os testes, vá na pasta **testes**, abra o terminal dela e execute:
```bash
pytest test_model.py
```

## FRONT-END

- Não é necessário executar nenhum comando para que o código funcione. Ao analisar o código, percebe-se que o CSS e o JavaScript estão embutidos, já que o projeto consiste em apenas uma página.
