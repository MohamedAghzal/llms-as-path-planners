import json
import heapq
import sys
import pandas as pd

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
def is_goal(grid, pos, actions, nx, ny):

    nx = len(grid)
    ny = len(grid[0])
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
            return -1
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

def get_metrics(data, org, nb_obstacles=None):

    ans = 0
    manhattan = 0
    invalid = 0
    em  = 0
    opt = 0
    off_grid = 0
    cnt = 0
    valid = 0

    tokens = 0
            
    no_goal = []
    no_go = []
    no_em = []
    unreach = 0
    for i in range(len(data)):

        grid = data[i]['world']
        
        cot = data[i]['predicted'].replace('.', '').replace('Therefore ', 'Therefore, ')
        print(cot)
        if('action sequence is: ' in cot):
            predicted = data[i]['predicted'].replace('.', '').replace('Therefore ', 'Therefore, ').split(': ')[1]
        else:
            predicted = 'Goal not reachable'

        #predicted = cot
        print(predicted)
        truth = data[i]['ground_truth']
        if('Goal not reachable' in truth):
            unreach += 1
            continue
        obsts = 0
        for k in range(len(grid)):
            for p in range(len(grid[0])):
                obsts += (grid[k][p] == 1)

        '''if(data[i]['ground_truth'] == 'Goal not reachable'):
            continue'''
        #obsts = len(data[i]['ground_truth'].split(' ')[:-1])
        if(nb_obstacles is None or obsts == nb_obstacles):
                print('Obsts', obsts, 'Nb Obsts', nb_obstacles)
                em += (truth.replace(' ', '') == predicted.replace(' ', ''))

                #init = (-1, -1)
                for k in range(len(grid)):
                    for p in range(len(grid[0])):
                        print(grid[k][p], end = ' ')
                        if(grid[k][p] == 2):
                            init = (k, p)
                    print()
                print()

                cnt += 1
                
                test = is_goal(grid, init, predicted, nx=20, ny=20)
                
                if(test == 1): 
                    ans += 1
                elif(test == 2):
                    off_grid += 1

                if(test != 1):
                    d = {}
                    for k in org[i]:
                        d[k] = org[i][k]
                    d['CoT'] = cot
                    no_goal.append(d)

                print('No Goal', len(no_goal))
                dist = distance_from_goal(grid, init, predicted, 20, 20)

                opt += is_optimal(grid, truth, init, predicted, 20, 20)
                if(dist == -1 or dist == 100):
                    invalid += 1
                elif(dist != -1 and test != 1 and not 'not reachable' in truth): 
                    print('Distance', dist)
                    manhattan += dist
                    valid += 1

                if(test != 1):
                    print({
                            'Utterance': data[i]['english'],
                            'Predicted':predicted,
                            'Ground Truth':truth
                        })
                    no_go.append(
                        {
                            'Utterance': data[i]['english'],
                            'Predicted':predicted,
                            'Ground Truth':truth
                        }
                    )
                if(truth.replace(' ', '') != predicted.replace(' ', '')):
                    no_em.append(
                        {
                            'Utterance': data[i]['english'],
                            'Predicted':predicted,
                            'Ground Truth':truth
                        }
                    )
    
    with open('no_em_naive.json', 'w') as f:
        ob = json.dumps(no_em)
        f.write(ob)
    
    with open('no_goal_naive.json', 'w') as f:
        ob = json.dumps(no_go)
        f.write(ob)  

    print(cnt, valid, ans, opt, manhattan, unreach, em)
    return {
        'Total': cnt,
        'EM': em / cnt,
        'Is Goal': ans / cnt,
        #'Distance': manhattan / valid,
        'Valid': 1 - invalid / cnt,
        'Optimal': opt / cnt,
        'Tokens': tokens,
        'Unreachable': unreach
    }, no_goal


f = open(sys.argv[1])
l = open(sys.argv[2])
data = json.load(f)
org = json.load(l)

metrics, path_length = get_metrics(data, org, None)

print(metrics)

'''
out_stats = []
for num in range(1, 6):
    print('Obsts', num)
    metrics, path_length = get_metrics(data, org, num)

    for metric in metrics:
        print(metric, metrics[metric])

    l = []
    for metric in metrics:
        l.append(metrics[metric])
    
    out_stats.append([num] + l[1:])

df = pd.DataFrame(out_stats, columns=['path-len', 'EM', 'Success rate', 'Valid', 'Optimal', 'Tokens'])
df.to_excel('per_obst_obst_cot_sg.xlsx', index=False)
'''

