import json
import numpy as np
import heapq

def a_star(grid, start, goal):
    grid = np.array(grid)
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
    
    print('This is the path', path)
    for i in range(len(path) - 1):
        curr = path[i]
        nxt = path[i + 1]

        print(curr)
        print(nxt)
        if(curr[0] + 1 == nxt[0]):
            directions += 'down '
        elif (curr[0] - 1 == nxt[0]):
            directions += 'up '
        elif (curr[1] + 1 == nxt[1]):
            directions += 'right '
        elif (curr[1] - 1 == nxt[1]):
            directions += 'left '

    return directions

def solution_plan(plan):

    path = ''
    for i in range(len(plan)):
        curr = plan[i][0]
        nxt = plan[i][1]

        sub_path = solution_point(plan[i])
        path += sub_path + 'inspect '

    return path


def extract_goals(nl_description):
    
    goals = {}

    positions = nl_description.split('.')[4].split('p')
    
    for e in positions:
        
        if(e == ' '):
            continue
            
        sp = e.split(' is located at ')
        
        goal_num = sp[0].replace(' ', '')
        location = sp[1].replace(' ', '').replace('),', '').replace(')', '').replace('(','').replace('and','').split(',')
        
        goals[int(goal_num)] = (int(location[0]), int(location[1]))
            
    return goals


def parse_order(order):
    out = order.split(',')
    locs = []

    for i in range(len(order) - 1):
        if(order[i] == 'p'):
            locs.append(int(order[i+1]))

    return locs

f = open('outputs/GPT-4-outputs-opt-ordering.json')

data = json.load(f)

outputs = []

for i in range (len(data)):
    instance = data[i]

    nl = instance['english']

    locations = []
    order = instance['predicted'].split(':')[1]
    print(order)

    predicted_order = parse_order(order)

    goals = extract_goals(nl)
    
    for loc in predicted_order:
        locations.append(goals[loc])
        print(loc)
        print(loc, goals[loc])

    
    world = instance['world']

    pos = (-1, -1)
    for i in range(len(world)):
        for j in range(len(world[i])):
            if(world[i][j] == 2):
                pos = (i, j)

    locations = [pos] + locations

    path = []

    for i in range(len(locations) - 1):
        sub = a_star(world, locations[i], locations[i+1])
        path.append(sub)

    print(path)
    plan = solution_plan(path)

    ins = {}

    for j in instance.keys():
        ins[j] = instance[j]

    ins['full_path'] = plan

    outputs.append(ins)

with open('ordering_check.json', 'w') as f:
    ob = json.dumps(outputs, indent = 4)
    f.write(ob)
    


