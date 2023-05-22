import heapq

def astar(graph, start_node, goal_node, heuristic):
    # Create a priority queue to store nodes with their associated costs
    priority_queue = [(0, start_node)]  # Tuple: (priority, node)
    # Create a dictionary to keep track of the cost to reach each node
    cost_so_far = {start_node: 0}
    # Create a dictionary to keep track of the path
    path = {start_node: None}

    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)

        if current_node == goal_node:
            return reconstruct_path(path, goal_node)

        # Explore the neighbors of the current node
        for neighbor, cost in graph[current_node].items():
            new_cost = cost_so_far[current_node] + cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic[neighbor]
                path[neighbor] = current_node
                heapq.heappush(priority_queue, (priority, neighbor))

    # If there's no path from the start to the goal node
    return None

def reconstruct_path(path, goal_node):
    # Reconstruct the path from the goal node to the start node
    current_node = goal_node
    path_list = [current_node]

    while path[current_node] is not None:
        current_node = path[current_node]
        path_list.append(current_node)

    path_list.reverse()
    return path_list

# Example usage
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8},
    'D': {'B': 5, 'C': 8}
}

heuristic = {
    'A': 7,
    'B': 2,
    'C': 6,
    'D': 0
}

start_node = 'A'
goal_node = 'D'

path = astar(graph, start_node, goal_node, heuristic)
print(path)
