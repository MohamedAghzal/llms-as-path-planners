import json
import heapq

def is_goal(grid, pos, actions, nx, ny):

    if(pos == (-1, -1)):
        return 1

    actions = actions.replace('move ', 'move_').replace('turn ', 'turn_').replace('moves ', 'move_')
    orientation = 'south'

    sequence = actions.lower().split(' ')

    moves = {
        'east': (0, 1), 
        'west': (0, -1), 
        'south': (1, 0), 
        'north': (-1, 0)
    }

    orient_left = {
        'south': 'east',
        'north': 'west',
        'west' : 'south',
        'east' : 'north'
    }

    orient_right = {
        'south': 'west',
        'north': 'east',
        'west' : 'north',
        'east' : 'south'
    }
    for action in sequence:
        x = pos[0]
        y = pos[1]
        if(action == 'turn_left'):
            orientation = orient_left[orientation]
        if(action == 'turn_right'):
            orientation = orient_right[orientation]
        if(action == 'move_forward'):
            pos = (x + moves[orientation][0], y + moves[orientation][1])

        if(pos[0] < 0 or pos[1] < 0 or pos[0] >= nx or pos[1] >= ny):
            return 0
        if(grid[pos[0]][pos[1]] == 1):
            return 0
    
    #print(pos[0], pos[1])
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

    return 1000

def distance_from_goal(grid, pos, actions, nx, ny):
    
    if(pos == (-1, -1)):
        return 1

    goal = ()
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if(grid[i][j] == 3):
                goal = (i, j)

    actions = actions.replace('move ', 'move_').replace('moves ', 'move_').replace('turn ', 'turn_')
    orientation = 'south'

    sequence = actions.lower().split(' ')

    moves = {
        'east': (0, 1), 
        'west': (0, -1), 
        'south': (1, 0), 
        'north': (-1, 0)
    }

    orient_left = {
        'south': 'east',
        'north': 'west',
        'west' : 'south',
        'east' : 'north'
    }

    orient_right = {
        'south': 'west',
        'north': 'east',
        'west' : 'north',
        'east' : 'south'
    }

    for action in sequence:
        x = pos[0]
        y = pos[1]
        if(action == 'turn_left'):
            orientation = orient_left[orientation]
        if(action == 'turn_right'):
            orientation = orient_right[orientation]
        if(action == 'move_forward'):
            pos = (x + moves[orientation][0], y + moves[orientation][1])

        if(pos[0] < 0 or pos[1] < 0 or pos[0] >= nx or pos[1] >= ny):
            return 100
        if(grid[pos[0]][pos[1]] == 1):
            return -1
    
    #print(pos[0], pos[1])
    if(grid[pos[0]][pos[1]] == 3):
        return 0
    else: 
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

    for i in range(len(data)):
        
        grid = data_original[i]['world']
        predicted = data[i]['generated'][0].replace('move ', 'move_').replace('turn ', 'turn_')
        truth = data[i]['ground_truth'].replace('move ', 'move_').replace('turn ', 'turn_')

        obsts = 0
        for k in range(len(grid)):
            for p in range(len(grid[0])):
                obsts += (grid[k][p] == 1)
        
        if(nb_obstacles is None or obsts == nb_obstacles):
                
                em += (truth.replace(' ', '') == predicted.replace(' ', ''))

                for k in range(len(grid)):
                    for p in range(len(grid[0])):
                        print(grid[k][p], end = ' ')
                        if(grid[k][p] == 2):
                            init = (k, p)
                    print()
                print()

                cnt += 1
                
                test = is_goal(grid, init, predicted, 5, 5)
                if(test == 1): 
                    ans += 1
                elif(test == 2):
                    off_grid += 1
                    
                dist = distance_from_goal(grid, init, predicted, 5, 5)

                opt += is_optimal(grid, truth, init, predicted, 5, 5)
                if(dist == -1 or dist == 100):
                    invalid += 1
                else: manhattan += dist

                if(truth.replace(' ', '') == predicted.replace(' ', '') and test != 1):
                    print('OOO', truth, predicted)
    
    return {
        'Total': cnt,
        'EM': em / cnt,
        'Is Goal': ans / cnt,
        'Distance': manhattan / cnt,
        'Valid': 1 - invalid / cnt,
        'Optimal': opt / cnt
    }



f = open('out_T5_base_sg_direction_5x5_unseen.json')
data = json.load(f)

f2 = open('../single_goal/5x5worlds/5x5worlds_samples.json')
data_original = json.load(f2)

metrics = get_metrics(data, data_original, None)

for metric in metrics:
    print(metric, metrics[metric])
