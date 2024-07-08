from generate_samples import a_star_value
import numpy as np
import sys
import json
from representations import generate_grid

def exact_match(correct, predicted):
    return correct.replace(' ', '') == predicted.replace(' ', '')

def valid_position(pos, grid):
    return pos[0] >= 0 and pos[0] < len(grid) and pos[1] >= 0 and pos[1] < len(grid[0]) and grid[pos[0]][pos[1]] != 1

def success_sg(grid, predicted, start=(-1, -1)):
    path = predicted.split()

    if(start == (-1, -1)):
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if(grid[i][j] == 2):
                    start = (i, j)
    
    moves = {
        'up': (-1, 0), 
        'down': (1, 0), 
        'left': (0, -1), 
        'right': (0, 1)
    }

    valid = False

    curr = start
    for step in path:
        if(step not in ['left', 'right', 'up', 'down']):
            continue
        if(step == 'left'):
            new_curr = (curr[0], curr[1] - 1)
        elif(step == 'right'):
            new_curr = (curr[0], curr[1] + 1)
        elif(step == 'up'):
            new_curr = (curr[0] - 1, curr[1])
        elif(step == 'down'):
            new_curr = (curr[0] + 1, curr[1])

        curr = new_curr
            
    
        if(valid_position(curr, grid) and grid[curr[0]][curr[1]] == 3):
            valid = True
            return valid
        if(not valid_position(curr, grid)):
            break

    return valid

def optimal(correct, predicted):
    return success_sg(correct, predicted) and len(correct.split(' ')) == len(predicted.split(' '))

def grid2path(grid_str, org_grid):

    grid_spt = grid_str.split('\n')

    grid = []
    for line in grid_spt:
        l = line.split()
        grid.append([int(x) for x in l])
    
    for i in range(len(org_grid)):
        for j in range(len(org_grid[0])):
            if(org_grid[i][j] == 2):
                start = (i, j)
    
    seen = set()

    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]

    dir = {
        (0, 1): 'right',
        (0, -1): 'left',
        (1, 0): 'down',
        (-1, 0): 'up'
    }


    while(True):
        path = ''
        adjacent = False
        for i in range(4):
            step = (start[0] + dx[i], start[1] + dy[i])
            if(not valid_position(step, grid)):
                continue
            if(grid[step[0]][step[1]] == 4):
                path += dir[step] + ' '
                start = step
                adjacent = True
                break
        
        if(not adjacent):
            break

    return path
    
def success_grid(grid, grid_str):
    path = grid2path(grid_str, org_grid=grid)
    return success_sg(grid, path)

def distance_to_goal(grid, predicted):

    path = predicted.split()

  
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if(grid[i][j] == 2):
                start = (i, j)
            if(grid[i][j] == 3):
                goal = (i, j)

    valid = False

    curr = start
    for step in path:
        if(step not in ['left', 'right', 'up', 'down']):
            continue
        if(step == 'left'):
            new_curr = (curr[0], curr[1] - 1)
        elif(step == 'right'):
            new_curr = (curr[0], curr[1] + 1)
        elif(step == 'up'):
            new_curr = (curr[0] - 1, curr[1])
        elif(step == 'down'):
            new_curr = (curr[0] + 1, curr[1])

        if(not valid_position(new_curr, grid) or grid[new_curr[0]][new_curr[1]] == 1):
            break

        curr = new_curr
         
        if(grid[curr[0]][curr[1]] == 3):
            return 0
        

    res = 0
    if(not valid):
        res = a_star_value(np.array(grid), curr, goal)
        print(res)
    if(res == 100000):
        print(generate_grid(grid))
        print(curr, goal)
        res = 0
    
    return res    
    

def main():
    choice = sys.argv[1]

    if(choice == 'distance'):
        iid_file = json.load(open(sys.argv[2]))
        ood_file = json.load(open(sys.argv[3]))

        iid_scores = []
        ood_scores = []
        if('react' in sys.argv[2]):
            for i in range(len(iid_file)):
                for out in iid_file[i]['Outputs']:
                    world = out['world']
                    path = ''
                    for msg in out['messages']:
                        if(msg['role'] == 'assistant'):
                            path += msg['content'] + ' '
                    dist = distance_to_goal(world, path)
                    if(dist == 0):
                        continue
                    iid_scores.append(dist)
            
            for i in range(len(ood_file)):
                for out in ood_file[i]['Outputs']:
                    world = out['world']
                    path = ''
                    for msg in out['messages']:
                        if(msg['role'] == 'assistant'):
                            path += msg['content'] + ' '
                    dist = distance_to_goal(world, path)
                    if(dist == 0):
                        continue
                    ood_scores.append(dist)
            
            print('Average distance I.I.D', np.mean(iid_scores))
            print('Average distance O.O.D', np.mean(ood_scores))
        elif('decomp' in sys.argv[2]):
            for i in range(len(iid_file)):
                for k in iid_file[i]:
                    path = ''
                    cnt = 0
                    curr = 0
                    #print(len(k))
                    for out in k:
                        world = out['World']
                        path += out['Predicted']
                        dist = distance_to_goal(world, path)
                        if(dist == 0):
                            continue
                        curr += dist
                        cnt += 1
                    if(cnt == 0):
                        continue
                    curr /= cnt
                    iid_scores.append(curr)
            
            for i in range(len(ood_file)):
                for k in ood_file[i]:
                    path = ''
                    cnt = 0
                    curr = 0
                    #print(len(k))
                    for out in k:
                        world = out['World']
                        path += out['Predicted']
                        dist = distance_to_goal(world, path)
                        if(dist == 0):
                            continue
                        curr += dist
                        cnt += 1
                    if(cnt == 0):
                        continue
                    curr /= cnt
                    ood_scores.append(curr)
            
            print('Average distance I.I.D', np.mean(iid_scores))
            print('Average distance O.O.D', np.mean(ood_scores))

if __name__ == "__main__":
    main()       
        
    