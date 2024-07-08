from planning_samples import geometry_samples
import geometries
import place_agent_goals_sg
from prompting import VALS_IID, VALS_OOD
import copy
from planning_samples import geometry_samples
from generate_samples import solution_point
from evaluate import success_sg

def fixed_length(iid_samples, ood_samples, values, n_instances):

    to_keep_iid = []
    to_keep_ood = []

    f = ood_samples
    f2 = iid_samples

    lengths = {}

    envs = []

    for i in range(200):
        envs.append([])

    for x in f:
        l = len(x['path'].split())

        envs[l].append(str(x['world']).replace('2', '0').replace('3', '0'))
        if(l not in lengths):
            lengths[l] = 0
        lengths[l] += 1

    for i in range(len(f2)):
        world = str(f2[i]['world']).replace('2', '0').replace('3', '0')
        valid = True

        for val in values:
            if(world not in envs[val]):
                valid = False
        if(valid):
            to_keep_iid.append(f2[i])

    for i in range(len(f)):
        world = str(f[i]['world']).replace('2', '0').replace('3', '0')
        valid = True

        for val in values:
            if(world not in envs[val]):
                valid = False
        if(valid):
            to_keep_ood.append(f[i])
    
    return {
        'IID': to_keep_iid[:n_instances], 
        'OOD': to_keep_ood[:n_instances],
        'Train': to_keep_iid[n_instances:]
    }

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

def single_environment(iid_samples, ood_samples, n_instances, geometry='rectangles'):

    to_keep_iid = []
    to_keep_ood = []

    envs = {}

    for i in range(len(iid_samples)):
        world = str(iid_samples[i]['world']).replace('2', '0').replace('3', '0')
        if(world not in envs.keys()):
            envs[world] = []
        else:
            envs[world].append(iid_samples[i])
    
    print('IID', len(iid_samples))
    return {
        'IID': iid_samples[:n_instances], 
        'OOD': ood_samples[:n_instances],
        'Train': iid_samples[n_instances:]
    }

def single_environment_random(n_instances, geometry='rectangles'): #generate environments on the fly
    
    env = geometries.sample(25, geometry, 1)
    data = place_agent_goals_sg.build_environments(env, low=2, high=26, vals = VALS_IID[geometry], trials=10)
    iid_samples = geometry_samples(data, shape=geometry)

    data = place_agent_goals_sg.build_environments(env, low=26, high=140, vals = VALS_OOD[geometry], trials=10)
    ood_samples = geometry_samples(data, shape=geometry)

    to_keep_iid = []
    to_keep_ood = []

    envs = {}

    for i in range(len(iid_samples)):
        world = str(iid_samples[i]['world']).replace('2', '0').replace('3', '0')
        if(world not in envs.keys()):
            envs[world] = []
        else:
            envs[world].append(iid_samples[i])
    
    return {
        'IID': iid_samples[:n_instances], 
        'OOD': ood_samples[:n_instances],
        'Train': iid_samples[n_instances:]
    }

def decompose_sample(sample, max_len=5, geometry='rectangle'):
    path = sample['path'].split()

    grid = copy.deepcopy(sample['world'])
    goal = sample['goals'][0]

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if(grid[i][j] == 2):
                start = (i, j)

    
    coords = [start]
    curr = start
    for step in path:
        if(step == 'left'):
            new_curr = (curr[0], curr[1] - 1)
        elif(step == 'right'):
            new_curr = (curr[0], curr[1] + 1)
        elif(step == 'down'):
            new_curr = (curr[0] + 1, curr[1])
        elif(step == 'up'):
            new_curr = (curr[0] - 1, curr[1])
        else:
            continue
        coords.append(new_curr)
        curr = new_curr
    
    subproblems = []
    if(len(coords) <= max_len + 1):
        processed_subproblems = [sample]
    else:
        offset = 0
        while(True):
            curr_subset = []
            l = offset
            r = 1 + offset
            print(l, r)

            curr_subset.append(coords[l])
 
            while(r - l <= max_len and r < len(coords)):
                curr_subset.append(coords[r])
                r += 1
            offset = r - 1

            new_world = copy.deepcopy(grid)

            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if(new_world[i][j] in [2, 3]):
                        new_world[i][j] = 0

            new_world[curr_subset[0][0]][curr_subset[0][1]] = 2
            new_world[curr_subset[-1][0]][curr_subset[-1][1]] = 3

            new_sample = {
                'world': new_world,
                'path': solution_point(curr_subset),
                'start': curr_subset[0],
                'goals': [curr_subset[-1]],
                'obstacles': sample['obstacles']
            }  

            if('points' in sample.keys()):
                new_sample['points'] = sample['points']
            
            subproblems.append(new_sample)

            if(goal in curr_subset or r >= len(coords)):
                break

        processed_subproblems = geometry_samples(subproblems, shape=geometry) 

    return processed_subproblems, subproblems


def unsuccessful_samples(file):
    unsuccessful = []
    for inst in file:
        for k in inst:
            predicted = k['Predicted']
            world = k['World']

            if(not success_sg(world, predicted)):
                unsuccessful.append(k)

    return unsuccessful