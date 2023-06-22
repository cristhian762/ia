import random

def crossoverOx(parentA, parentB):
    cutPoints = sorted(random.sample(range(len(parentA)), 2))

    child1 = parentA[cutPoints[0]:cutPoints[1]]
    child2 = parentB[cutPoints[0]:cutPoints[1]]

    for gene in parentB:
        if gene not in child1:
            child1.append(gene)
    for gene in parentA:
        if gene not in child2:
            child2.append(gene)

    return cutPoints, child1, child2

parentA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
parentB = [10, 3, 2, 9, 4, 8, 7, 6, 1, 5]

cutPoints, child1, child2 = crossoverOx(parentA, parentB)

print("Ponto de corte:", cutPoints)
print("Filho 1:", child1)
print("Filho 2:", child2)
