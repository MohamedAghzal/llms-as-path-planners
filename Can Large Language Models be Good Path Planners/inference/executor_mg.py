import json
import heapq
import pandas as pd
import sys

def extract_goals(nl_description):
    goals = []

    positions = nl_description.split('.')[4].split('p')

    for e in positions:
        
        if(e == ' '):
            continue
            
        sp = e.split(' is located at ')
        
        goal_num = sp[0].replace(' ', '')
        location = sp[1].replace(' ', '').replace('),', '').replace(')', '').replace('(','').replace('and','').split(',')
        
        goals.append((int(location[0]), int(location[1])))
        
    return goals



def check_constraint(actions, pos, goals, grid, nx=6, ny=6, constraint=None):
    '''
        -1: Constraints not satisfied and not all goals visited.
        0: All goals visited but constraints not satisfied
        1: Valid solution
        -2: Invalid Path (Obstacle or Off-Grid)
    '''
    acts = ['up', 'down', 'left', 'right']

    nx = len(grid)
    ny = len(grid[0])

    sequence = actions.split(' ')

    numbers = {}

    for i in range(len(goals)):
        numbers[goals[i]] = i

    inspected = []
    for action in sequence:
        x = pos[0]
        y = pos[1]
        
        if(action == 'left'):
            pos = (x, y-1)
        if(action == 'right'):
            pos = (x, y+1)
        if(action == 'down'):
            pos = (x+1, y)
        if(action == 'up'):
            pos = (x-1, y)
        if(action == 'inspect' and pos in goals):
            print(pos, 'inspected')
            inspected.append(pos)
        
        if(pos[0] < 0 or pos[1] < 0 or pos[0] >= nx or pos[1] >= ny or grid[pos[0]][pos[1]] == 1):
            return -2, [], pos
            break
    
    visited = {}

    for i in range(len(goals)):
        visited[numbers[goals[i]]] = 0

    ans = 0
    for i in range(len(inspected)):
        curr = numbers[inspected[i]]
        visited[curr] = 1

    if(constraint is not None and 'before' in constraint):
        plan = constraint.split('before')
        first = plan[0]
        last = plan[1]

        before = []
        for i in range(len(first)):
            if(first[i] == 'p'):
                before.append(int(first[i+1]))

        after = []
        for i in range(len(last)):
            if(last[i] == 'p'):
                after.append(int(last[i+1]))

        for i in range(len(inspected)):
            curr = numbers[inspected[i]]


            if(curr in after):
                for k in before:
                    if(visited[k] == 0 and ans != -2):
                        ans = -1     

    valid = True
    unvisited = []
    for i in range(len(goals)):
        if(visited[numbers[goals[i]]] != 1):
            valid = False
            unvisited.append(goals[i])
    
    print('Visited list', visited, goals)
    if(ans == -2):
        return -2, unvisited, pos
    if(valid and ans == -1):
        return 0, [], pos
    elif(len(unvisited) > 0):
        return -1, unvisited, pos
    return 1, [], pos

def a_star(grid, start, goal):
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(position):
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

    visited = set()
    heap = []
    heapq.heappush(heap, (0, start, []))

    while heap:
        cost, current, path = heapq.heappop(heap)
        if grid[current[0]][current[1]] == 3:
            return len(path + [current])

        if current in visited:
            continue

        visited.add(current)

        for action in actions:
            neighbor = (current[0] + action[0], current[1] + action[1])

            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[1]):
                if grid[neighbor[0]][neighbor[1]] != 1:
                    new_cost = cost + 1
                    new_path = path + [current]
                    heapq.heappush(heap, (new_cost + heuristic(neighbor), neighbor, new_path))

    return 100

def distance_from_goal(grid, pos, unvisited, nx=6, ny=6):

    dist = 0

    for city in unvisited:
        dist += a_star(grid, pos, city)

    return dist

def is_optimal(ground_truth, actions):

    return (len(actions.replace(' ', '')) <= len(ground_truth.replace(' ', '')))

def get_metrics(data, data_original, nb_obstacles=None):

    ans = 0
    manhattan = 0
    invalid = 0
    em  = 0
    opt = 0
    all_visited = 0
    cnt = 0
    denom = 0
    for i in range(len(data)):
        #print(data[i]['predicted'])
        grid = data_original[i]['world']
        predicted = data[i]['generated'][0]
        truth = data[i]['ground_truth']
        nl_description = data[i]['english'] 
        goals = extract_goals(nl_description)

        if(truth == ''):
            continue
        obsts = 0
        for k in range(len(grid)):
            for p in range(len(grid[0])):
                obsts += (grid[k][p] == 1)
        
        #obsts = len(truth.split(' ')[:-1])
        if(nb_obstacles is None or obsts == nb_obstacles):

                
                if('before' not in nl_description):
                    continue
                

                em += (truth.replace(' ', '') == predicted.replace(' ', ''))

                
                for k in range(len(grid)):
                    for p in range(len(grid[0])):
                        print(grid[k][p], end = ' ')
                        if(grid[k][p] == 2):
                            init = (k, p)
                    print()
                print()

                cnt += 1
                
                constraint = None

                if('before' in nl_description):
                    constraint = nl_description.split('Visit ')[1]
                    print('Constr.', constraint)
                
                test, unvisited, pos = check_constraint(predicted, init, goals, grid, constraint=constraint)

                if(test == 1): 
                    ans += 1
                elif(test == 0):
                    all_visited += 1
                elif(test == -1):
                    dist = distance_from_goal(grid, init, unvisited, 6, 6)
                    print(dist)
                    if(dist > 0 and dist < 100): 
                        manhattan += dist
                        denom += 1
                elif(test == -2):
                    invalid += 1

                if(test != 1 and test != 0 and (truth.replace(' ', '') == predicted.replace(' ', ''))):
                    print('oho', predicted, unvisited)

                opt += (test == 1 and is_optimal(truth, predicted))


    print(manhattan, denom)
    return {
        'Total': cnt,
        'EM': em / cnt,
        'Success rate': ans / cnt,
       # 'Distance': manhattan / denom,
        'Valid': 1 - invalid / cnt,
        'All Visited': all_visited / cnt,
        'Optimal': opt / cnt
    }

f = open(sys.argv[1])
data = json.load(f)

f2 = open(sys.argv[2])
data_original = json.load(f2)


if(sys.argv[3] == 'None'):
    n_obs = None
else:
    n_obs = int(sys.argv[3])

out_stats = []

metrics = get_metrics(data, data_original, None)

for metric in metrics:
    print(metric, metrics[metric])

df = pd.DataFrame(out_stats, columns=['path-len', 'EM', 'Success rate', 'Valid', 'All Visited', 'Optimal'])
df.to_excel('per_obst_stats_moreobsts_react_mg.xlsx', index=False)
    
