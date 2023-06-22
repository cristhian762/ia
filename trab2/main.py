# Implementar: 

# [HC-R] – Hill-Climbing with Restart;
# [SA]   – Simulated Annealing e
# [GA]   – Genetic Algorithm.

# Rodar 10 vezes

# (1) TSP (Travelling Salesman Problem – Problema do Cacheiro-Viajante);
# (2) Minimização de Função Objetivo univariada;
# (3) Problema das 8-rainhas (8-queens Problem).

import numpy as np

def generateRandomCoords(nCities):
    min = 10
    max = 90
    scale = (max - min) - 1

    X = min + scale * np.random.rand(nCities)


def main():

    nCities = 10

    defineCoords = generateRandomCoords(nCities)

    print(defineCoords)