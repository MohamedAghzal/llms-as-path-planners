import numpy as np 
import heapq

def generate_grid(grid):
    grid_str = ''

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            grid_str += f'{grid[i][j]} '
        grid_str += '\n'

    return grid_str

def generate_code(x, y, obstacles, goals, initial_loc):

    output = f"""
 
obstacles = []
goals = {goals}
initial_location = {initial_loc}
"""

    per_line_obsts = []
    for i in range(x):
        obsts_in_row = []
        free_from_obsts = []
        for obst in obstacles:
            if(obst[0] == i):
                obsts_in_row.append(obst[1])

        for j in range(y):
            if(j not in obsts_in_row):
                free_from_obsts.append(j)

        if(len(obsts_in_row) >= x / 2):
            fors = f"""
for j in range({x}):
    if(j not in {free_from_obsts}): 
        obstacles.append([{i}, j]) 
"""
        elif(len(obsts_in_row) > 0):
            fors = f"""
for j in range({x}):
    if(j in {per_line_obsts}):
       obstacles.append([{i}, j]) 
"""
        

        per_line_obsts.append(obsts_in_row)

        if(len(obsts_in_row)): 
            output += fors
        

    return output

def a_star(grid, start, goal):
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(position):
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

    visited = set()
    heap = []
    heapq.heappush(heap, (0, start, []))
    while heap:
        cost, current, path = heapq.heappop(heap)

        if current == goal:
            return path + [current]

        if current in visited:
            continue

        visited.add(current)

        for action in actions:
            neighbor = (current[0] + action[0], current[1] + action[1])

            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor] != 1:
                    new_cost = cost + 1
                    new_path = path + [current]
                    heapq.heappush(heap, (new_cost + heuristic(neighbor), neighbor, new_path))

    return 'Goal not reachable'

def a_star_value(grid, start, goal):
        grid = np.array(grid)
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def heuristic(position):
            return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

        visited = set()
        heap = []
        heapq.heappush(heap, (0, start, []))

        while heap:
            cost, current, path = heapq.heappop(heap)

            current = tuple(current)
            if current == goal:
                return len(path + [current])

            if current in visited:
                continue

            visited.add(current)

            for action in actions:
                neighbor = (current[0] + action[0], current[1] + action[1])

                if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor] != 1:
                        new_cost = cost + 1
                        new_path = path + [current]
                        heapq.heappush(heap, (new_cost + heuristic(neighbor), neighbor, new_path))

        return 100000

def solution_point(path):

    if(path == 'Goal not reachable'):
        return path

    directions = ''
    
    for i in range(len(path) - 1):
        curr = path[i]
        nxt = path[i + 1]

        if(curr[0] + 1 == nxt[0]):
            directions += 'down '
        elif (curr[0] - 1 == nxt[0]):
            directions += 'up '
        elif (curr[1] + 1 == nxt[1]):
            directions += 'right '
        elif (curr[1] - 1 == nxt[1]):
            directions += 'left '

    return directions



    