# Implementar: 

# [HC-R] – Hill-Climbing with Restart;
# [SA]   – Simulated Annealing e
# [GA]   – Genetic Algorithm.

# Rodar 10 vezes

# (1) TSP (Travelling Salesman Problem – Problema do Cacheiro-Viajante);
# (2) Minimização de Função Objetivo univariada;
# (3) Problema das 8-rainhas (8-queens Problem).
import pickle

import random

import numpy as np
import pandas as pd

import math

import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def generateDataFrameCost(algorithmData):
    names  = algorithmData.keys()

    lin = len(names)
    col = 10

    dataFrameCost = pd.DataFrame(np.zeros((lin, col)), index=names)

    dataFrameCost.index.name='ALGORITMO'

    for algorithm, item in algorithmData.items():
        for i in range(10):
            dataFrameCost.loc[algorithm, i] = item[i]['cost']

    return dataFrameCost

def linearRankSelection(population):
    populationSize = len(population)
    rankScores = [i+1 for i in range(populationSize)]  # Pontuações de classificação linear
    totalScore = sum(rankScores)

    selectionProbabilities = [score / totalScore for score in rankScores]

    selectedParents = []

    while len(selectedParents) < 2:  # Selecionar dois pais
        rand = random.random()
        cumulativeProbability = 0.0
        for i, probability in enumerate(selectionProbabilities):
            cumulativeProbability += probability
            if rand <= cumulativeProbability:
                selectedParents.append(population[i])
                break

    return selectedParents

def crossover(parent1, parent2):
    N = len(parent1)
    offspring = [None] * N

    start, end = sorted(random.sample(range(N), 2))

    offspring[start:end+1] = parent1[start:end+1]

    for i in range(N):
        if offspring[i] is None:
            j = 0
            while parent2[j] in offspring:
                j += 1
            offspring[i] = parent2[j]

    return offspring

def mutate(tsp, solution, mutationRate):
    N = len(solution)
    
    for i in range(N):
        if random.random() < mutationRate:
            j = random.randint(0, N-1)
            solution[i], solution[j] = solution[j], solution[i]
    
    return solution

def geneticAlgorithm(tsp, populationSize, numGenerations, mutationRate):
    population = generatePopulation(tsp, populationSize)

    for _ in range(numGenerations):
        population = evolvePopulation(tsp, population, mutationRate)

    bestSolution = population[0]
    bestCost = calculateCost(tsp, bestSolution)

    return bestCost, bestSolution

def generatePopulation(tsp, populationSize):
    population = []

    for _ in range(populationSize):
        solution = randomResult(tsp)
        population.append(solution)

    return population

def evolvePopulation(tsp, population, mutationRate):
    newPopulation = []

    # Elitismo: mantém a melhor solução da geração anterior
    bestSolution = population[0]
    newPopulation.append(bestSolution)

    # Geração de novas soluções através de cruzamento e mutação
    while len(newPopulation) < len(population):
        parent1, parent2 = linearRankSelection(population)  # Seleção por classificação linear

        offspring = crossover(parent1, parent2)
        offspring = mutate(tsp, offspring, mutationRate)

        newPopulation.append(offspring)

    return newPopulation

def simulatedAnnealing(tsp, initSolution, initialTemperature, coolingRate):
    currentSolution = initSolution
    currentCost = calculateCost(tsp, currentSolution)

    bestSolution = currentSolution
    bestCost = currentCost

    temperature = initialTemperature

    maxIterations = int(0.9 * (len(tsp) * len(tsp)))  # Definindo o número máximo de iterações

    for iteration in range(maxIterations):
        neighborSolution = generateNeighbor(currentSolution)
        neighborCost = calculateCost(tsp, neighborSolution)

        if neighborCost < currentCost:
            currentSolution = neighborSolution
            currentCost = neighborCost
        else:
            acceptProb = acceptanceProbability(currentCost, neighborCost, temperature)

            # Reduzindo a probabilidade linearmente
            if random.random() < acceptProb:
                currentSolution = neighborSolution
                currentCost = neighborCost

        if currentCost < bestCost:
            bestSolution = currentSolution
            bestCost = currentCost

        # Reduzindo a temperatura linearmente
        temperature *= coolingRate

    return bestCost, bestSolution

def acceptanceProbability(currentCost, neighborCost, temperature):
    if neighborCost < currentCost:
        return 1.0
    else:
        return math.exp((currentCost - neighborCost) / temperature)

def generateNeighbor(solution):
    neighbor = solution.copy()
    N = len(solution)
    i = random.randint(1, N-1)
    j = random.randint(1, N-1)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor

def hillClimbingWithRestart(tsp, numRestarts):
    bestSoluction = None
    bestCost = float('inf')

    for _ in range(numRestarts):
        initSolution = randomResult(tsp)

        soluction, cost = hillClimbing(tsp, initSolution)

        if cost < bestCost:
            bestCost = cost
            bestSoluction = soluction

    return bestCost, bestSoluction

def plotRoutes(dataFrameCities, soluction):
    dataFrameSoluction = dataFrameCities.copy()
    dataFrameSoluction = dataFrameSoluction.reindex(soluction)

    X = dataFrameSoluction['X']
    Y = dataFrameSoluction['Y']

    cities = list(dataFrameSoluction.index)

    fig = go.Figure()

    fig.update_layout(autosize=False, width=500, height=500, showlegend=False)

    fig.add_trace(go.Scatter(x=X, y=Y, text=cities, textposition='bottom center', mode='lines+markers+text', name=''))

    fig.add_trace(go.Scatter(x=X.iloc[[-1,0]], y=Y.iloc[[-1,0]], mode='lines+markers', name=''))

    fig.show()

def boxplotSorted(df, rot=90, figsize=(12,6), fontsize=20):
    df2 = df.T
    meds = df2.median().sort_values(ascending=False)
    axes = df2[meds.index].boxplot(
        figsize      = figsize,
        rot          = rot, 
        fontsize     = fontsize, 
        boxprops     = dict(linewidth=4, color='cornflowerblue'), 
        whiskerprops = dict(linewidth=4, color='cornflowerblue'),
        medianprops  = dict(linewidth=4, color='firebrick'), 
        capprops     = dict(linewidth=4, color='cornflowerblue'), 
        flierprops   = dict(marker='o', markerfacecolor='dimgray', markersize=12, markeredgecolor='black'), 
        return_type  = "axes"
    )

    axes.set_title("Cost of Algorithms", fontsize=fontsize)

def generateNeighbors(solution):
    N = len(solution)
    for i in range(1, N):
        for j in range(i + 1, N):
            vizinho = solution.copy()
            vizinho[i] = solution[j]
            vizinho[j] = solution[i]

            yield(vizinho)

def calculateCost(tsp, solution):
    N = len(solution)
    cost = 0

    for i in range(N):
        k = (i+1) % N
        cityA = solution[i]
        cityB = solution[k]

        cost += tsp.loc[cityA, cityB]

    return cost

def bestNeighbor(tsp, solution):
    bestCost = calculateCost(tsp, solution)
    bestNeighbor = solution

    for neighbor in generateNeighbors(solution):
        cost = calculateCost(tsp, neighbor)
        if cost < bestCost:
            bestCost = cost
            bestNeighbor = neighbor

    return bestNeighbor, bestCost

def randomResult(tsp):
    result = []
    
    cities = list(tsp.keys())

    result.append(cities.pop(0))

    for _ in range(0,len(cities)):
        city = random.choice(cities)

        result.append(city)
        cities.remove(city)

    return result

def hillClimbing(tsp, soluction):
    bestSoluction, bestCost = bestNeighbor(tsp, soluction)

    while True:
        soluction, cost = bestNeighbor(tsp, bestSoluction)

        if cost < bestCost:
            bestCost   = cost
            bestSoluction = soluction
        else:
            break

    return bestSoluction, bestCost

def calculateEuclideanDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def generateDistanceMatrix(dataFrameCities):
    nCities = len(dataFrameCities)

    dist = np.zeros((nCities,nCities), dtype=float)

    for i in range(0, nCities):
        for j in range(i+1, nCities):
            x1,y1 = dataFrameCities.iloc[i]
            x2,y2 = dataFrameCities.iloc[j]

            dist[i,j] = calculateEuclideanDistance(x1, y1, x2, y2)
            dist[j,i] = dist[i,j]

    return dist

def generateTSP(dataFrameCities):
    cities = dataFrameCities.index
    
    dists = generateDistanceMatrix(dataFrameCities)

    return pd.DataFrame(dists, columns=cities, index=cities)

def generateRandomCoords(nCities):
    min = 10
    max = 90
    scale = (max - min) - 1

    X = min + scale * np.random.rand(nCities)
    Y = min + scale * np.random.rand(nCities)

    coords = {'X':X, 'Y':Y}

    cities = ['A'+str(i) for i in range(nCities)]

    dataFrameCities = pd.DataFrame(coords, index=cities)
    dataFrameCities.index.name = 'CIDADE'

    return dataFrameCities

def generateResultTSP(saveData=False):
    nCities = 10

    dataFrameCities = generateRandomCoords(nCities)

    tsp = generateTSP(dataFrameCities)

    # file = open("dataFrameCities", "wb")
    # pickle.dump(dataFrameCities, file)
    # file.close

    # file = open("tsp", "wb")
    # pickle.dump(tsp, file)
    # file.close

    # with open('dataFrameCities', 'rb') as f:
    #     dataFrameCities = pickle.load(f)

    # with open('tsp', 'rb') as f:
    #     tsp = pickle.load(f)

    result = {
        'random': [],
        'hillClimbing': [],
        'simulatedAnnealing': [],
        'geneticAlgorithm': [],
    }

    for _ in range(10):
        randomSolution = randomResult(tsp)
        randomCost = calculateCost(tsp, randomSolution)
        
        dicRandom = {
            'cost': randomCost,
            'solution': randomSolution
        }

        # plotRoutes(dataFrameCities, randomSolution)
        # print(randomCost)

        cost_hc, soluction_hc = hillClimbingWithRestart(tsp, 30)

        dicHillClimbing = {
            'cost': cost_hc,
            'solution': soluction_hc
        }

        # plotRoutes(dataFrameCities, soluction_hc)
        # print(cost_hc)

        initialTemperature = 100.0
        coolingRate = 0.99

        cost_sa, soluction_sa = simulatedAnnealing(tsp, randomResult(tsp), initialTemperature, coolingRate)

        dicSimulatedAnnealing = {
            'cost': cost_sa,
            'solution': soluction_sa
        }

        # plotRoutes(dataFrameCities, soluction_sa)
        # print(cost_sa)

        mutationRate = 0.1
        populationSize = 50
        numGenerations = 70

        cost_ga, soluction_ga = geneticAlgorithm(tsp, populationSize, numGenerations, mutationRate)
        
        dicGeneticAlgorithm = {
            'cost': cost_ga,
            'solution': soluction_ga
        }

        # plotRoutes(dataFrameCities, soluction_ga)
        # print(cost_ga)

        result['random'].append(dicRandom)
        result['hillClimbing'].append(dicHillClimbing)
        result['simulatedAnnealing'].append(dicSimulatedAnnealing)
        result['geneticAlgorithm'].append(dicGeneticAlgorithm)
    
    data = {
        'result': result,
        'dataFrameCities': dataFrameCities
    }

    if(saveData):
        file = open('result', 'wb')
        pickle.dump(data, file)
        file.close

    return data

def getData():
    with open('result', 'rb') as f:
        data = pickle.load(f)

    return data

def main():
    # data = generateResultTSP()

    data = getData()
    print(data)

    dataFrameCost = generateDataFrameCost(data['result'])
    print(dataFrameCost)

    boxplotSorted(dataFrameCost, rot=90, figsize=(12,6), fontsize=20)
    print(dataFrameCost.describe())

    print(data['result'])
    for algorithm, item in data['result'].items():
        bestSoluctin = item[0]['solution']
        bestCost = item[0]['cost']

        for i in range(1, 10):
            if item[i]['cost'] < bestCost:
                bestCost = item[i]['cost']
                bestSoluctin = item[i]['solution']
        
        print(algorithm)
        print(bestCost)
        plotRoutes(data['dataFrameCities'], bestSoluctin)

if __name__ == "__main__":
    main()