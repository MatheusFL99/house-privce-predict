import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def prever_nova_casa(modelo, area, qualidade, ano, garagem, porao): # Faz uma previsão do preço de uma nova casa com base nas características fornecidas.
    nova = np.array([[area, qualidade, ano, garagem, porao]])
    return modelo.predict(nova)[0]


def avaliar_modelo(y_true, y_pred): # Calcula e imprime as métricas de desempenho do modelo.
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'MSE (Erro Quadrático Médio): {mse:.2f}')
    print(f'R² (Coeficiente de Determinação): {r2:.2f}')
    return mse, r2


def plotar_resultados(y_true, y_pred): # Plota os valores reais vs previstos.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("Preço Real")
    plt.ylabel("Preço Previsto")
    plt.title("Preço Real vs. Preço Previsto")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.grid(True)
    plt.show()
