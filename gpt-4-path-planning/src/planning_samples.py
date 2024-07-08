import representations
import place_agent_goals_sg
import one_entrance
import geometries
from generate_samples import a_star, solution_point
import json
import numpy as np
import random

def zig_zag_samples(data, axis = 'row'):

    samples = []
    for world in data:
        grid = world['world']

        obstacles = [[obs[0], obs[1]] for obs in world['obstacles']]

        start = world['start']
        goals = world['goals']


        nl = representations.naive_enumeration(len(grid), len(grid[0]), obstacles, goals, start)

        grid_rep = representations.generate_grid(grid)

        world['grid_representation'] = grid_rep
        world['naive_representation'] = nl
        
        if(axis == 'row'):
            world['code_representation'] = representations.generate_code_alternate_row(25, 25, obstacles, goals, start)
        else:
            world['code_representation'] = representations.generate_code_alternate_column(25, 25, obstacles, goals, start)
        
        if(len(goals) == 1):
            coordinates = a_star(np.array(grid), (start[0], start[1]), (goals[0][0], goals[0][1]))
            sol_point = solution_point(coordinates)

            world['path'] = sol_point
            
            samples.append(world)

    return samples

def is_row(world):
    row_is_filled = False

    for i in range(len(world)):
        for j in range(len(world[0])):
            if(j > 0 and world[i][j] == 1 and world[i][j - 1] == 1):
                row_is_filled = True
    
    return row_is_filled

def geometry_samples(data, shape='rectangle'):
    samples = []
    for world in data:
        grid = world['world']

        obstacles = [[obs[0], obs[1]] for obs in world['obstacles']]

        start = world['start']
        goals = world['goals']


        nl = representations.naive_enumeration(len(grid), len(grid[0]), obstacles, goals, start)

        grid_rep = representations.generate_grid(grid)

        world['grid_representation'] = grid_rep
        world['naive_representation'] = nl
        if(shape == 'rectangle'):
            world['code_representation'] = representations.generate_code_rectangle(
                x=25, 
                y=25, 
                goals=goals, 
                obstacles=obstacles, 
                initial_loc=start, 
                points=world['points']
            )
        elif(shape == 'triangle'):
            world['code_representation'] = representations.generate_code_triangle(
                x=25,
                y=25,
                obstacles=obstacles,
                goals=goals,
                initial_loc=start
            )

        elif(shape == 'maze'):
            world['code_representation'] = representations.generate_code_spiral(
                size_x=25,
                size_y=25,
                obstacles=obstacles,
                goals=goals,
                initial_loc=start
            )
        elif(shape == 'zig_zag'):
            check = is_row(grid)
            if(check):
                world['code_representation'] = representations.generate_code_alternate_row(
                    25, 
                    25, 
                    obstacles, 
                    goals, 
                    start
                )
            else:
                world['code_representation'] = representations.generate_code_alternate_column(
                    25, 
                    25, 
                    obstacles, 
                    goals, 
                    start
                )

        if(len(goals) == 1):
            coordinates = a_star(np.array(grid), (start[0], start[1]), (goals[0][0], goals[0][1]))
            sol_point = solution_point(coordinates)
            world['path'] = sol_point
            samples.append(world)
    
    return samples

def main():

    #IID
    triangs = geometries.sample(25, 'maze', 200)
    data_triangs = place_agent_goals_sg.build_environments(triangs, low=2, high=26, vals = [5, 10, 15, 20, 25], trials=3)

    sampled_tri = geometry_samples(data_triangs, shape='maze')

    with open('maze_data_sg_iid.json', 'w') as f:
        ob = json.dumps(sampled_tri, indent=4)
        f.write(ob)

    data_triangs = place_agent_goals_sg.build_environments(triangs, low=26, high=200, vals = [30, 40, 50, 60, 75], trials=1)

    sampled_tri = geometry_samples(data_triangs, shape='maze')

    with open('maze_data_sg_ood.json', 'w') as f:
        ob = json.dumps(sampled_tri, indent=4)
        f.write(ob)

    X = one_entrance.sample(25, 'row', 200)
    Y = one_entrance.sample(25, 'column', 200)

    #in distribution
    data_one_entrance_row = place_agent_goals_sg.build_environments(X, low=2, high=26, vals = [5, 10, 15, 20, 25], trials=3)
    data_one_entrance_col = place_agent_goals_sg.build_environments(Y, low=2, high=26, vals = [5, 10, 15, 20, 25], trials=3)

    r = zig_zag_samples(data_one_entrance_row, axis='row')
    c = zig_zag_samples(data_one_entrance_col, axis='column')

    
    one_entrance_data = r + c

    #random.shuffle(one_entrance_data)

    with open('zig_zag_data_sg_indist.json', 'w') as f:
        ob = json.dumps(one_entrance_data, indent=4)
        f.write(ob)

    ''' 
    X = one_entrance.sample(25, 'row', 50)
    Y = one_entrance.sample(25, 'column', 50)
    '''

    #out of distribution
    data_one_entrance_row = place_agent_goals_sg.build_environments(X, low=26, high=200, vals = [30, 50, 60, 75, 100], trials=1)
    data_one_entrance_col = place_agent_goals_sg.build_environments(Y, low=26, high=200, vals = [30, 50, 60, 75, 100], trials=1)

    r = zig_zag_samples(data_one_entrance_row, axis='row')
    c = zig_zag_samples(data_one_entrance_col, axis='column')

    one_entrance_data = r + c

    random.shuffle(one_entrance_data)
    
    with open('zig_zag_data_sg_ood.json', 'w') as f:
        ob = json.dumps(one_entrance_data, indent=4)
        f.write(ob)

    rects = geometries.sample(25, 'rectangle', 200)
    data_rects = place_agent_goals_sg.build_environments(rects, low=2, high=26, vals=[2, 5, 10, 15, 20], trials=3)

    sampled_rect = geometry_samples(data_rects, shape='rectangle')

    with open('rectangle_data_sg_iid.json', 'w') as f:
        ob = json.dumps(sampled_rect, indent=4)
        f.write(ob)
    
    # OOD
    data_rects = place_agent_goals_sg.build_environments(rects, low=26, high=200, vals = [25, 30, 35, 40, 45], trials=1)

    sampled_rect = geometry_samples(data_rects, shape='rectangle')

    with open('rectangle_data_sg_ood.json', 'w') as f:
        ob = json.dumps(sampled_rect, indent=4)
        f.write(ob)

    '''
    rects = geometries.sample(25, 'random', 200)
    data_rects = place_agent_goals_sg.build_environments(rects, low=2, high=26, vals=[2, 5, 10, 15, 20])

    sampled_rect = geometry_samples(data_rects, shape='random')

    with open('random_data_sg_iid.json', 'w') as f:
        ob = json.dumps(sampled_rect, indent=4)
        f.write(ob)

    data_rects = place_agent_goals_sg.build_environments(rects, low=2, high=26, vals=[25, 30, 35, 40, 45])

    sampled_rect = geometry_samples(data_rects, shape='random')

    with open('random_data_sg_ood.json', 'w') as f:
        ob = json.dumps(sampled_rect, indent=4)
        f.write(ob)
    '''
if __name__ == '__main__':
    main()
 