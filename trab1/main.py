import time
import random
import heapq

from collections import deque
# from viewer import MazeViewer
from math import inf, sqrt



def gera_labirinto(n_linhas, n_colunas, inicio, goal):
    # cria labirinto vazio
    labirinto = [[0] * n_colunas for _ in range(n_linhas)]

    numero_de_obstaculos = int(0.5 * n_linhas * n_colunas / 2)
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


def esta_contido(lista, celula):
    for elemento in lista:
        if (elemento.y == celula.y) and (elemento.x == celula.x):
            return True
    return False


def custo_caminho(caminho):
    if len(caminho) == 0:
        return inf

    custo_total = 0
    for i in range(1, len(caminho)):
        custo_total += distancia(caminho[i].anterior, caminho[i])

    return custo_total


def obtem_caminho(goal):
    caminho = []

    celula_atual = goal
    while celula_atual is not None:
        caminho.append(celula_atual)
        celula_atual = celula_atual.anterior

    # o caminho foi gerado do final para o
    # comeco, entao precisamos inverter.
    caminho.reverse()

    return caminho

def distancia(celula_1, celula_2):
    dx = celula_1.x - celula_2.x
    dy = celula_1.y - celula_2.y

    return sqrt(dx ** 2 + dy ** 2)

def celulas_vizinhas_livres(celula_atual, labirinto):
    # generate neighbors of the current state
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

    # seleciona as celulas livres
    vizinhos_livres = []
    for v in vizinhos:
        # verifica se a celula esta dentro dos limites do labirinto
        if (v.y < 0) or (v.x < 0) or (v.y >= len(labirinto)) or (v.x >= len(labirinto[0])):
            continue
        # verifica se a celula esta livre de obstaculos.
        if labirinto[v.y][v.x] == 0:
            vizinhos_livres.append(v)

    return vizinhos_livres

def cost_to_neighbor(current_node, neighbor):
    return 1 # Assume custo uniforme

def breadth_first_search(labirinto, inicio, goal, viewer):
    # nos gerados e que podem ser expandidos (vermelhos)
    fronteira = deque()
    # nos ja expandidos (amarelos)
    expandidos = set()

    # adiciona o no inicial na fronteira
    fronteira.append(inicio)

    # variavel para armazenar o goal quando ele for encontrado.
    goal_encontrado = None

    # Repete enquanto nos nao encontramos o goal e ainda
    # existem para serem expandidos na fronteira. Se
    # acabarem os nos da fronteira antes do goal ser encontrado,
    # entao ele nao eh alcancavel.
    while (len(fronteira) > 0) and (goal_encontrado is None):

        # seleciona o no mais antigo para ser expandido
        no_atual = fronteira.popleft()

        # busca os vizinhos do no
        vizinhos = celulas_vizinhas_livres(no_atual, labirinto)

        # para cada vizinho verifica se eh o goal e adiciona na
        # fronteira se ainda nao foi expandido e nao esta na fronteira
        for v in vizinhos:
            if v.y == goal.y and v.x == goal.x:
                goal_encontrado = v
                # encerra o loop interno
                break
            else:
                if (not esta_contido(expandidos, v)) and (not esta_contido(fronteira, v)):
                    fronteira.append(v)

        expandidos.add(no_atual)

        #viewer.update(generated=fronteira,
        #             expanded=expandidos)

    caminho = obtem_caminho(goal_encontrado)
    custo   = custo_caminho(caminho)

    return caminho, custo, expandidos


def depth_first_search(labirinto, inicio, goal, viewer):
    fronteira = deque()
    
    expandidos = set()
    
    fronteira.append(inicio)
    
    goal_encontrado = None
    
    while (len(fronteira) > 0) and (goal_encontrado is None):
        no_atual = fronteira.pop()
    
        vizinhos = celulas_vizinhas_livres(no_atual, labirinto)
    
        for v in vizinhos:
            if v.y == goal.y and v.x == goal.x:
                goal_encontrado = v
                break
            else:
                if (not esta_contido(expandidos, v)) and (not esta_contido(fronteira, v)):
                    fronteira.append(v)
    
        expandidos.add(no_atual)
    
        # viewer.update(generated=fronteira,
        #               expanded=expandidos)

    caminho = obtem_caminho(goal_encontrado)
    custo   = custo_caminho(caminho)

    return caminho, custo, expandidos
        
def a_star_search(labirinto, inicio, goal, viewer):
    expandidos = set()

    priority_queue = [(0, inicio)]
    cost_so_far = {inicio: 0}
    path = {inicio: None}

    goal_encontrado = None

    while (len(priority_queue) > 0 and (goal_encontrado is None)):
        _, current_node = heapq.heappop(priority_queue)

        if current_node.x == goal.x and current_node.y == goal.y:
            goal_encontrado = current_node
            break

        neighbors = celulas_vizinhas_livres(current_node, labirinto)
        for neighbor in neighbors:
            new_cost = cost_so_far[current_node] + cost_to_neighbor(current_node, neighbor)

            if neighbor not in cost_so_far  or new_cost < cost_so_far[neighbor]:
                if (not esta_contido(expandidos, neighbor)):
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + distancia(neighbor, goal)
                    path[neighbor] = current_node
                
                    heapq.heappush(priority_queue, (priority, neighbor))

        expandidos.add(current_node)

        caminho = obtem_caminho(current_node)

        # viewer.update(generated=[node for _, node in priority_queue],
        #             expanded=expandidos)

    caminho = obtem_caminho(goal_encontrado)
    custo   = custo_caminho(caminho)

    return caminho, custo, expandidos

def uniform_cost_search(labirinto, inicio, goal, viewer):
    expandidos = set()

    priority_queue = [(0, inicio)]
    cost_so_far = {inicio: 0}
    path = {inicio: None}

    goal_encontrado = None

    while (len(priority_queue) > 0 and (goal_encontrado is None)):
        _, current_node = heapq.heappop(priority_queue)

        if current_node.x == goal.x and current_node.y == goal.y:
            goal_encontrado = current_node
            break

        neighbors = celulas_vizinhas_livres(current_node, labirinto)
        for neighbor in neighbors:
            if neighbor not in cost_so_far:
                if (not esta_contido(expandidos, neighbor)):
                    priority = distancia(neighbor, goal)
                    path[neighbor] = current_node
                
                    heapq.heappush(priority_queue, (priority, neighbor))

        expandidos.add(current_node)

        caminho = obtem_caminho(current_node)

        # viewer.update(generated=[node for _, node in priority_queue],
        #               expanded=expandidos)

    caminho = obtem_caminho(goal_encontrado)
    custo   = custo_caminho(caminho)

    return caminho, custo, expandidos

#-------------------------------


def main():
    SEED = 314  # coloque None no lugar do 42 para deixar aleatorio
    random.seed(SEED)
    N_LINHAS  = 200
    N_COLUNAS = 300 
    INICIO = Celula(y=0, x=0, anterior=None)
    GOAL   = Celula(y=N_LINHAS-1, x=N_COLUNAS-1, anterior=None)


    """
    O labirinto sera representado por uma matriz (lista de listas)
    em que uma posicao tem 0 se ela eh livre e 1 se ela esta ocupada.
    """
    labirinto = gera_labirinto(N_LINHAS, N_COLUNAS, INICIO, GOAL)

    # viewer = MazeViewer(labirinto, INICIO, GOAL,
    #                     step_time_miliseconds=20, zoom=40)

    #----------------------------------------
    # BFS Search
    #----------------------------------------
    # viewer._figname = "BFS"
    start = time.time()
    caminho, custo_total, expandidos = \
            breadth_first_search(labirinto, INICIO, GOAL, None)
    end = time.time()
    
    if len(caminho) == 0:
        print("Goal é inalcançavel neste labirinto.")

    print(
        f"BFS:"
        f"\tTempo: {(end - start)}.\n"
        f"\tNumero de nos expandidos: {len(expandidos)}.\n"
        f"\tNumero de nos gerados: {N_LINHAS * N_COLUNAS / 2}.\n"
        f"\tCusto do caminho: {custo_total}.\n"
        f"\tTamanho do caminho: {len(caminho)-1}.\n\n"
    )

    # viewer.update(path=caminho)
    # viewer.pause()


    #----------------------------------------
    # DFS Search
    #----------------------------------------
    # viewer._figname = "DFS"
    start = time.time()
    caminho, custo_total, expandidos = \
            depth_first_search(labirinto, INICIO, GOAL, None)
    end = time.time()

    if len(caminho) == 0:
        print("Goal é inalcançavel neste labirinto.")

    print(
        f"DFS:"
        f"\tTempo: {(end - start)}.\n"
        f"\tNumero de nos expandidos: {len(expandidos)}.\n"
        f"\tNumero de nos gerados: {N_LINHAS * N_COLUNAS / 2}.\n"
        f"\tCusto do caminho: {custo_total}.\n"
        f"\tTamanho do caminho: {len(caminho)-1}.\n\n"
    )

    # viewer.update(path=caminho)
    # viewer.pause()

    #----------------------------------------
    # A-Star Search
    #----------------------------------------
    # viewer._figname = "A*"
    start = time.time()
    caminho, custo_total, expandidos = \
            a_star_search(labirinto, INICIO, GOAL, None)
    end = time.time()

    if len(caminho) == 0:
        print("Goal é inalcançavel neste labirinto.")

    print(
        f"A*:"
        f"\tTempo: {(end - start)}.\n"
        f"\tNumero de nos expandidos: {len(expandidos)}.\n"
        f"\tNumero de nos gerados: {N_LINHAS * N_COLUNAS / 2}.\n"
        f"\tCusto do caminho: {custo_total}.\n"
        f"\tTamanho do caminho: {len(caminho)-1}.\n\n"
    )

    # viewer.update(path=caminho)
    # viewer.pause()

    #----------------------------------------
    # Uniform Cost Search (Obs: opcional)
    #----------------------------------------

    # viewer._figname = "Uniform Cost Search"
    start = time.time()
    caminho, custo_total, expandidos = \
            uniform_cost_search(labirinto, INICIO, GOAL, None)
    end = time.time()

    if len(caminho) == 0:
        print("Goal é inalcançavel neste labirinto.")

    print(
        f"UCS:"
        f"\tTempo: {(end - start)}.\n"
        f"\tNumero de nos expandidos: {len(expandidos)}.\n"
        f"\tNumero de nos gerados: {N_LINHAS * N_COLUNAS / 2}.\n"
        f"\tCusto do caminho: {custo_total}.\n"
        f"\tTamanho do caminho: {len(caminho)-1}.\n\n"
    )

    # viewer.update(path=caminho)
    # viewer.pause()

    print("OK! Pressione alguma tecla pra finalizar...")
    input()


if __name__ == "__main__":
    main()
