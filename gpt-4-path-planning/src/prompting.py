import json
import random
from representations import generate_grid
import copy
import sys

VALS_IID = {
    'rectangle': [2, 5, 10, 15, 20],
    'random': [2, 5, 10, 15, 20],
    'zig_zag': [5, 10, 15, 20, 25],
    'maze': [5, 10, 15, 20, 25]
}

VALS_OOD = {
    'rectangle': [25, 30, 35, 40, 45],
    'random': [25, 30, 35, 40, 45],
    'zig_zag':  [30, 50, 60, 75, 100],
    'maze': [30, 40, 50, 60, 75]
}

def out_of_bounds(grid, position):
    return position[0] < 0 or position[1] < 0 or position[0] >= len(grid) or position[1] >= len(grid[0])

def draw_path(org_grid, path, init):
    steps = path.split()

    grid = org_grid.copy()
    for i in range(len(steps)):
        if(out_of_bounds(grid, init) or grid[init[0]][init[1]] == 1):
            return grid
        grid[init[0]][init[1]] = 4
        if(steps[i] == 'left'):
            init = (init[0], init[1] - 1)
        if(steps[i] == 'right'):
            init = (init[0], init[1] + 1)
        if(steps[i] == 'up'):
            init = (init[0] - 1, init[1])
        if(steps[i] == 'down'):
            init = (init[0] + 1, init[1])
    
    return grid

def generate_effects(sample, start, goal):
    world = sample['world']

    path = sample['path'].split()
    thought = ''

    curr = start
    for i in range(len(path)):
        options = {
            'left': (0, -1),
            'right': (0, 1),
            'up': (-1, 0),
            'down': (1, 0)
        }

        thought += f'I am now at {curr}. '
        for option in options.keys():
            new_curr = (curr[0] + options[option][0], curr[1] + options[option][1])
            if(out_of_bounds(world, new_curr)):
                thought += f'Going {option} would lead me outside the grid. '
            elif(world[new_curr[0]][new_curr[1]] == 1):
                thought += f'Going {option} would lead me to the obstacle at {new_curr}. '
            elif(path[i] == option):
                thought += f'I can go {path[i]}. '
                curr = new_curr
            if(new_curr == goal):
                thought += 'Goal Reached!'
                break
    
    return thought
                

def n_shot_prompt(samples, n_exemplars, lengths = None, AE_type=None):

    if(lengths == None):
        examples = random.sample(samples, n_exemplars)
    else:
        if(len(lengths) > n_exemplars):
            print('Not enough examples to generate all lengths..\n Generating few shot examples of random lengths')
            examples = random.sample(samples, n_exemplars)
        else:
            choices = {}

            for l in lengths:
                choices[l] = []

            for k in range(len(samples)):
                path_l = len(samples[k]['path'].split())
                if(path_l in lengths):
                    choices[path_l].append(samples[k])
            
            factor = n_exemplars // len(lengths)

            examples = []
            for val in lengths:
                ex = random.sample(choices[val], factor)
                for x in ex:
                    examples.append(x)


    prompt = f'''Generate a path to navigate from the initial location to the goal location similarly to the examples below. (0,0) is located in the upper-left corner and (M, N) lies in the M row and N column.\n '''

    p_naive = ''
    p_code = ''
    p_grid = ' 2 denotes the starting location, 3 denotes the goal location, while 1\'s denotes obstacles.'
    p_g2g = ' 2 denotes the starting location, 3 denotes the goal location, while 1\'s denotes obstacles. The path is drawn using 4\'s.' 
    p_ae = ''
    for ex in examples:
        p_naive += f'''
###
Task description: {ex['naive_representation']}
Solution: {ex['path']}
'''
        p_code += f'''
###
Description in code:
{ex['code_representation']}
Solution: {ex['path']}
'''
        p_grid +=  f'''
###
Grid representation:
{ex['grid_representation']}
Solution: {ex['path']}''' 
        
        init = (-1, -1)
        for i in range(len(ex['world'])):
            for j in range(len(ex['world'][0])):
                if(ex['world'][i][j] == 2):
                    init = (i, j)
                if(ex['world'][i][j] == 3):
                    goal = (i, j)

        if(init == (-1, -1)):
            for i in range(len(ex['world'])):
                for j in range(len(ex['world'][0])):
                    print(ex['world'][i][j], end='')
                print('')

        p_g2g +=  f'''###
Grid representation:
{ex['grid_representation']}
Solution: {generate_grid(draw_path(copy.deepcopy(ex['world']), ex['path'], init))}
'''     
        if(AE_type == None):
            AE_type = 'naive'

        p_ae += f'''###
Task Description: {ex[f'{AE_type}_representation']}
Thought: {generate_effects(ex, init, goal)}
Solution: {ex['path']}
'''
    n_shot = {
        'Code': prompt + p_code,
        'Grid': prompt + p_grid,
        'Naive': prompt + p_naive,
        'Grid2Grid': prompt + p_g2g, 
        'AE': prompt + p_ae, 
        'Samples': examples
    }

    return n_shot

    
def next_example(ex, n_shot, representation='Naive'):
    '''
    representation:
        Code
        Grid
        Naive
    '''

    prompt = n_shot[representation]

    if(representation == 'Naive' or representation == 'AE_Naive'):
        prompt += f'''
###
Task description: {ex['naive_representation']}
Solution: 
'''
    elif ('Grid' in representation or representation == 'AE_Grid'):
        prompt += f'''
###
Grid representation:
{ex['grid_representation']}
Solution: 
'''
    elif (representation == 'Code' or representation=='AE_Code'):
        prompt += f'''
###
Description in code:
{ex['code_representation']}
Solution: 
'''    
     
    return prompt

def example_inference(train, test_set, n_exemplars, representation):
    type_ = None
    if('AE' in representation):
         type_ = representation.split('_')[1]
         representation = 'AE'

    n_shot = n_shot_prompt(train, n_exemplars, None, AE_type=type_)

    out = []
    for ex in test_set:
        prompt = next_example(ex, n_shot=n_shot, representation=representation) 
        return prompt

def group_environments(data, geometry): #Assumption: data includes IID + OOD instances
    ids = {}
    datapoints = {}
    
    iid_samples = []
    ood_samples = []

    counter = 0
    for instance in data:
        world = str(instance['world']).replace('2', '0').replace('3', '0')
        path = instance['path'].split()

        if(world not in ids):
            ids[world] = counter
            datapoints[counter] = {'IID': [], 'OOD': []}
            id_ = counter
            counter += 1
        else:
            id_ = ids[world]

        if(len(path) < VALS_OOD[geometry][0]):
            datapoints[id_]['IID'].append(instance)
        else:
            datapoints[id_]['OOD'].append(instance)



    return datapoints

def main():
    geometry = sys.argv[1]
    representation = sys.argv[2]
    iid_data = json.load(open(sys.argv[3]))
    ood_data = json.load(open(sys.argv[4]))
    grouped =  group_environments(data=iid_data + ood_data, geometry=geometry)  
    valid = {}
    for id_ in grouped.keys():
        if(len(grouped[id_]['OOD']) < 5):
            continue
        valid[id_] = grouped[id_]

            
    for id_ in list(valid.keys())[:30]:
        print(f'Processing environment {id_}')
        test_samples = 5

        iid_values = VALS_IID[geometry]
        ood_values = VALS_OOD[geometry]

        count = {}
        
        test_iid = []
        for x in grouped[id_]['IID']:
            vv = len(x['path'].split())
            if(vv in count.keys()):
                continue
            count[vv] = True
            test_iid.append(x)

        train = []

        for x in grouped[id_]['IID']:
            if(x not in test_iid):
                train.append(x)
        
        test_ood = grouped[id_]['OOD'] 
        with open(f'../prompts-examples/{geometry}_{representation}.txt', 'w') as f:
            f.write(example_inference(train, test_ood, 5, representation))

if(__name__ == '__main__'):
    main()