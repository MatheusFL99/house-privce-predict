import numpy as np

def prever_nova_casa(modelo, area, qualidade, ano, garagem, porao):
    nova = np.array([[area, qualidade, ano, garagem, porao]])
    return modelo.predict(nova)[0]
