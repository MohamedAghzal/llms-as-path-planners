import random
import heapq

def construct_grid(n, obstacles):
    '''
    0: empty cell
    1: obstacle
    2: start location
    3: goal
    '''
    grid = []

    for i in range(n):
        row = []
        for j in range(n):
            if([i, j] in obstacles):
                row.append(1)
            else:
                row.append(0)
        grid.append(row)
    
    return grid

def build_path(grid, start, desired_distance, obstacles):
    # A typical A* implementation, modified to search for a specific path distance
    def neighbors(grid, curr):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        neigh = []
        for dx, dy in directions:
            next_x, next_y = curr[0] + dx, curr[1] + dy
            if(0 <= next_x < len(grid) and 0 <= next_y < len(grid[0]) and [next_x, next_y] not in obstacles):
                neigh.append((next_x, next_y))
        return neigh
    
    def heuristic(goal, position):
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])
    # Priority queue for the frontier, with initial node
    frontier = [(0, start)]  # (priority, node)
    came_from = {start: None}
    cost_so_far = {start: 0}
    possible_goals = []

    while frontier:
        current_priority, current = heapq.heappop(frontier)

        # If the current node is at the desired distance, record it as a possible goal
        if cost_so_far[current] == desired_distance:
            possible_goals.append(current)
            continue  # Continue searching for more possible goals
        
        for next in neighbors(grid, current):  # Define how to find neighbors
            new_cost = cost_so_far[current] + 1  
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, start) 
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    return possible_goals

def generate_worlds(obstacles, n, n_goals, trials=30, low_bound=None, high_bound=None, vals = None):
    if low_bound is None or high_bound is None:
        raise ValueError("low_bound and high_bound must be set for this function to work.")

    worlds = []

    for _ in range(trials):
        if vals == None:
            traversed = range(low_bound, high_bound + 1, 5)
        else:
            traversed = vals

        for target_dist in traversed:
            successful_world = False
            attempts = 0  

            while not successful_world and attempts < 250: 
                attempts += 1
                grid = construct_grid(n, obstacles)
                
                while True:
                    agent_x, agent_y = random.randint(0, n-1), random.randint(0, n-1)
                    if [agent_x, agent_y] not in obstacles:
                        grid[agent_x][agent_y] = 2  
                        break

                goals = []
                att__ = 0
                while(len(goals) < n_goals and att__ < 150):
                    path = build_path(grid, (agent_x, agent_y), target_dist, obstacles)
                    att__ += 1

                    if(len(path) > 0):
                        random.shuffle(path)
                        goals.append(path[-1])
                        grid[path[-1][0]][path[-1][1]] = 3

                sample = {
                    'world': grid,
                    'obstacles': obstacles,
                    'start': [agent_x, agent_y],
                    'goals': goals
                }

                if(sample in worlds):
                    continue

                if(len(goals) == n_goals):
                    
                    successful_world = True
                    worlds.append({
                        'world': grid,
                        'obstacles': obstacles,
                        'start': [agent_x, agent_y],
                        'goals': goals
                    })
                else:
                    grid[agent_x][agent_y] = 0  

    return worlds

def build_environments(envs, low=None, high=None, vals=None, trials=2):
    '''
    CLA:
        directory/setting
        shortest path length
        longest path length
    '''

    worlds = []
    for inst in envs:
        obstacles = [[obs[0], obs[1]] for obs in inst['obstacles']]
        shape = inst['shape']
        

        combinations = generate_worlds(obstacles, shape[0], 1, trials=trials, low_bound=low, high_bound=high, vals=vals)
        for comb in combinations:
            if('points' in inst.keys()):
                comb['points'] = inst['points']
            worlds.append(comb)
        
    return worlds