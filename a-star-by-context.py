import random
from collections import deque
from math import inf, sqrt
from viewer import MazeViewer
import heapq

def gera_labirinto(n_linhas, n_colunas, inicio, goal):
    # cria labirinto vazio
    labirinto = [[0] * n_colunas for _ in range(n_linhas)]

    # adiciona celulas ocupadas em locais aleatorios de
    # forma que 25% do labirinto esteja ocupado
    numero_de_obstaculos = int(0.50 * n_linhas * n_colunas)
    for _ in range(numero_de_obstaculos):
        linha = random.randint(0, n_linhas-1)
        coluna = random.randint(0, n_colunas-1)
        labirinto[linha][coluna] = 1

    # remove eventuais obstaculos adicionados na posicao
    # inicial e no goal
    labirinto[inicio.y][inicio.x] = 0
    labirinto[goal.y][goal.x] = 0

    return labirinto


class Celula:
    def __init__(self, y, x, anterior):
        self.y = y
        self.x = x
        self.anterior = anterior

    def __lt__(self, other):
        return False  # Nós não serão comparados, apenas usaremos a prioridade da fila de prioridade


def distancia(celula_1, celula_2):
    dx = celula_1.x - celula_2.x
    dy = celula_1.y - celula_2.y
    return sqrt(dx ** 2 + dy ** 2)


def celulas_vizinhas_livres(celula_atual, labirinto):
    vizinhos = [
        Celula(y=celula_atual.y-1, x=celula_atual.x-1, anterior=celula_atual),
        Celula(y=celula_atual.y+0, x=celula_atual.x-1, anterior=celula_atual),
        Celula(y=celula_atual.y+1, x=celula_atual.x-1, anterior=celula_atual),
        Celula(y=celula_atual.y-1, x=celula_atual.x+0, anterior=celula_atual),
        Celula(y=celula_atual.y+1, x=celula_atual.x+0, anterior=celula_atual),
        Celula(y=celula_atual.y+1, x=celula_atual.x+1, anterior=celula_atual),
        Celula(y=celula_atual.y+0, x=celula_atual.x+1, anterior=celula_atual),
        Celula(y=celula_atual.y-1, x=celula_atual.x+1, anterior=celula_atual),
    ]

    vizinhos_livres = []
    for v in vizinhos:
        if (v.y < 0) or (v.x < 0) or (v.y >= len(labirinto)) or (v.x >= len(labirinto[0])):
            continue  # Fora dos limites do labirinto

        if labirinto[v.y][v.x] == 0:
            vizinhos_livres.append(v)

    return vizinhos_livres


def a_star_search(labirinto, inicio, goal, viewer=None):
    priority_queue = [(0, inicio)]
    cost_so_far = {inicio: 0}
    path = {inicio: None}

    while priority_queue:
        _, current_node = heapq.heappop(priority_queue)

        if current_node == goal:
            break

        neighbors = celulas_vizinhas_livres(current_node, labirinto)
        for neighbor in neighbors:
            new_cost = cost_so_far[current_node] + 1  # Assume custo uniforme

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + distancia(neighbor, goal)
                heapq.heappush(priority_queue, (priority, neighbor))
                path[neighbor] = current_node

        if viewer:
            viewer.update(generated=[node for _, node in priority_queue],
                          expanded=list(cost_so_far.keys()))

    if goal not in path:
        return None, inf, cost_so_far.keys()

    path_nodes = []
    current_node = goal
    while current_node is not None:
        path_nodes.append(current_node)
        current_node = path[current_node]
    path_nodes.reverse()

    cost = cost_so_far[goal]

    return path_nodes, cost, cost_so_far.keys()


def main():
    while True:
        N_LINHAS = 10
        N_COLUNAS = 20
        INICIO = Celula(y=0, x=0, anterior=None)
        GOAL = Celula(y=N_LINHAS-1, x=N_COLUNAS-1, anterior=None)

        labirinto = gera_labirinto(N_LINHAS, N_COLUNAS, INICIO, GOAL)

        viewer = MazeViewer(labirinto, INICIO, GOAL, step_time_miliseconds=20, zoom=40)

        viewer._figname = "A*"
        caminho, custo_total, expandidos = a_star_search(labirinto, INICIO, GOAL, viewer)

        if not caminho:
            print("Goal é inalcançável neste labirinto.")
            break

        print(f"A*:\n\tCusto total do caminho: {custo_total}.\n\tNúmero de passos: {len(caminho) - 1}.\n\tNúmero total de nós expandidos: {len(expandidos)}.\n\n")

        viewer.update(path=caminho)

        print("Pressione alguma tecla para finalizar...")
        input()


if __name__ == "__main__":
    main()
