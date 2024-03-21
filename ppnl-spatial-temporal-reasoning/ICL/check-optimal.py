import json
import heapq
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import random

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

with open('ordering_check.json') as f:
    file = json.load(f)


optimal = 0
cnt = 0
for i in file:
    if('before' not in i['english']):
        continue
    a = i['full_path'].replace(' ', '')
    b = i['ground_truth'].replace(' ', '')

    if(i['ground_truth'] == ''):
        continue
    cnt += 1
    if(len(a) == len(b)):
        optimal += 1

print(cnt, optimal / cnt)

def solution_point(path):

    if(path == 'Goal not reachable'):
        return path

    directions = ''
    
    for i in range(len(path) - 1):
        curr = path[i]
        nxt = path[i + 1]

        print(curr, nxt)

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
        curr = plan[i]

        sub_path = solution_point(plan[i])
        path += sub_path + 'inspect '

    print('The path is ', path)
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

optimal = []
predicted = []


aggregated = json.load(open('outputs/react-mg.json'))

optimal = 0
count = 0
print(len(aggregated))
for i in range(len(file)):

    k = aggregated[i]
    p = file[i]
    locations = []

    if('before' not in  k['nl_description']): continue
    if(p['ground_truth'] == ''):
        continue


    predicted_order = k['predicted_order']
    goals = extract_goals(k['nl_description'])


    
    for loc in predicted_order:
        locations.append(goals[loc])
    
    path = []
    world = k['world']

    init = (-1, -1)
    for a in range(len(world)):
        for b in range(len(world[0])):
            if(world[a][b] == 2):
                init = (a, b)

    locations = [init] + locations
    print(len(locations), 'hhh')
    for j in range(len(locations)-1):
        sub = a_star(world, locations[j], locations[j+1])
        print('Subpath', sub)
        path.append(sub)

    print(path)
    plan = solution_plan(path)
    print(plan)
    count += 1

    print('LENGTHS', len(plan.replace(' ', '')), len(p['ground_truth'].replace(' ', '')))
    print('LENGTHS', path, plan,'----', p['ground_truth'])

    if(len(plan.replace(' ', '')) == len(p['ground_truth'].replace(' ', ''))):
        print('AHAAAA')
        optimal += 1


print(count, optimal / count)




