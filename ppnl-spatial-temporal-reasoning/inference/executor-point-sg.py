import json
import heapq
import pandas as pd 
import sys

x = int(sys.argv[3])
y = int(sys.argv[4])
def is_goal(grid, pos, actions, nx, ny):
    acts = ['up', 'down', 'left', 'right']

    for i in range(len(acts)):
        for j in range(len(acts)):
            actions = actions.replace(acts[i]+acts[j], acts[i]+ ' ' + acts[j])
    sequence = actions.split(' ')

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

        if(pos[0] < 0 or pos[1] < 0 or pos[0] >= nx or pos[1] >= ny):
            return 0
        elif(grid[pos[0]][pos[1]] == 1):
            return 2
    
    if(grid[pos[0]][pos[1]] == 3):
        return 1
    else: 
        return 0

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

def distance_from_goal(grid, pos, actions, nx, ny):
    goal = ()
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if(grid[i][j] == 3):
                goal = (i, j)

        sequence = actions.split(' ')

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

        if(pos[0] < 0 or pos[1] < 0 or pos[0] >= nx or pos[1] >= ny):
            return 100
        if(grid[pos[0]][pos[1]] == 1):
            return -1
    
    return a_star(grid, (pos[0], pos[1]), (goal[0], goal[1]))

def is_optimal(grid, ground_truth, pos, actions, nx, ny):

    return is_goal(grid, pos, actions, nx, ny) == 1 and (len(actions.replace(' ', '')) <= len(ground_truth.replace(' ', '')))

def get_metrics(data, data_original, nb_obstacles=None):

    ans = 0
    manhattan = 0
    invalid = 0
    em  = 0
    opt = 0
    off_grid = 0
    cnt = 0
    path_length = []
    valid = 0
    for i in range(len(data)):

        grid = data_original[i]['world']
        predicted = data[i]['generated'][0]
        truth = data[i]['ground_truth']
        if('Goal not reachable' in truth):
            continue
        obsts = 0
        for k in range(len(grid)):
            for p in range(len(grid[0])):
                obsts += (grid[k][p] == 1)
        
        print(obsts)



        if(nb_obstacles is None or obsts >= nb_obstacles):

                em += (truth.replace(' ', '') == data[i]['generated'][0].replace(' ', ''))

                
                for k in range(len(grid)):
                    for p in range(len(grid[0])):
                        print(grid[k][p], end = ' ')
                        if(grid[k][p] == 2):
                            init = (k, p)
                    print()
                print()

                cnt += 1
                
                test = is_goal(grid, init, predicted, x, y)
                if(test == 1): 
                    ans += 1
                elif(test == 2):
                    off_grid += 1

                print(truth.split(' '))
                pp = {
                    'n': len(truth.split(' ')[:-1]),
                    'exact_match':  (truth.replace(' ', '') == data[i]['generated'][0].replace(' ', '')) * 1,
                    'success': (test == 1) * 1,
                    'optimal': is_optimal(grid, truth, init, predicted, x, y) * 1,
                    'valid': (distance_from_goal(grid, init, predicted, x, y) != -1 
                            and distance_from_goal(grid, init, predicted, x, y) != 100) * 1
                }
                dist = distance_from_goal(grid, init, predicted, x, y)

                opt += is_optimal(grid, truth, init, predicted, x, y)

                if(dist == -1 or dist == 100):
                    invalid += 1
                elif(dist != -1): 
                    manhattan += dist
                    valid += 1

                path_length.append(pp)
        
    return {
        'Total': cnt,
        'EM': em / cnt,
        'Is Goal': ans / cnt,
        'Distance': manhattan / valid,
        'Valid': 1 - invalid / cnt,
        'Optimal': opt / cnt
    }, path_length

f = open(sys.argv[1])
data = json.load(f)

f2 = open(sys.argv[2])
data_original = json.load(f2)

print(len(data), len(data_original))

unique = []

for el in data:
    if(el in unique):
        continue
    unique.append(el)

data = unique

metrics, path_length = get_metrics(data, data_original, None)

for metric in metrics:
    print(metric, metrics[metric])

'''out_stats = []
for num in range(6, 12):
    metrics, path_length = get_metrics(data, data_original, num)

    for metric in metrics:
        print(metric, metrics[metric])

    l = []
    for metric in metrics:
        l.append(metrics[metric])
    
    out_stats.append([num] + l[1:])

df = pd.DataFrame(out_stats, columns=['path-len', 'EM', 'Success rate', 'Valid', 'All Visited', 'Optimal'])
df.to_excel('per_obst_stats_moreobsts_t5_sg.xlsx', index=False)
    


ns = {}
success = {}
em = {}
opti = {}
count = {}

for i in range(len(path_length)):
    ns[path_length[i]['n']] = 0
    success[path_length[i]['n']] = 0
    em[path_length[i]['n']] = 0
    opti[path_length[i]['n']] = 0
    count[path_length[i]['n']] = 0

for i in range(len(path_length)):
    ns[path_length[i]['n']] += path_length[i]['valid']
    success[path_length[i]['n']] += path_length[i]['success']
    em[path_length[i]['n']] += path_length[i]['exact_match']
    opti[path_length[i]['n']] += path_length[i]['optimal']
    count[path_length[i]['n']] += 1


outputs = []

for i in range(len(path_length)):
    res = {
        'N': path_length[i]['n'],
        'Success Rate': success[path_length[i]['n']] / count[path_length[i]['n']],
        'Exact Match': em[path_length[i]['n']] / count[path_length[i]['n']],
        'Valid': ns[path_length[i]['n']] / count[path_length[i]['n']],
        'Optimal': opti[path_length[i]['n']] / count[path_length[i]['n']]
    }

    if(res in outputs):
        continue

    outputs.append(res)

for out in outputs:
    print(out)

df = pd.DataFrame(outputs, columns = ['N', 'Success Rate', 'Exact Match', 'Valid', 'Optimal'])

print(df.head())

df.to_excel('length-stats.xlsx', index=False)'''