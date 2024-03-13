import random
import sys
import json
import os
def construct_grid(n, obstacles):
    '''
    0: empty cell
    1: obstacle
    2: start location
    3: goal
    '''
    grid = []

    for i in range(n):
        row = []
        for j in range(n):
            if([i, j] in obstacles):

                row.append(1)
            else:
                row.append(0)
        grid.append(row)
    
    return grid

def generate_worlds(obstacles, n, n_goals, trials=30):

    worlds = []
    for _ in range(trials):
        grid = construct_grid(n, obstacles)
        while(True):
            agent_x = random.randint(0, n-1)
            agent_y = random.randint(0, n-1)
            if([agent_x, agent_y] not in obstacles):
                break

        print(agent_x, agent_y)
        grid[agent_x][agent_y] = 2
    
        goals = []
        while(True):
            goal_x = random.randint(0, n-1)
            goal_y = random.randint(0, n-1)
            if([goal_x, goal_y] not in obstacles 
                    and [goal_x, goal_y] != [agent_x, agent_y]
                    and [goal_x, goal_y] not in goals):
                goals.append([goal_x, goal_y])
                grid[goal_x][goal_y] = 3
            
            if(len(goals) == n_goals):
                break

        worlds.append({
            'world': grid,
            'obstacles': obstacles,
            'start': [agent_x, agent_y],
            'goals': goals
        })

    return worlds


def main():
    '''
    CLA:
        directory/setting
        number of goals
    '''

    envs = os.listdir(str(sys.argv[1]))
    n_goals = int(sys.argv[2])
    
    worlds_train = []
    worlds_dev = []
    worlds_test = []

    
    for env in envs:
        with open(str(sys.argv[1]) + '/' + env) as f:
            data = json.load(f)

            train = 0.8 * len(data)
            test = 0.2 * len(data)

            for inst in data[:int(train)]:
                obstacles = inst['obstacles']
                shape = inst['shape']

                print(obstacles)
                combinations = generate_worlds(obstacles, shape[0], n_goals, trials=30)

                tr = int(0.8 * len(combinations))
                dv = int(0.1 * len(combinations))
                ts = int(0.1 * len(combinations))
                
                print(tr, dv, ts)
                print(len(combinations))
                for comb in combinations[:tr]:
                    worlds_train.append(comb)
                for comb in combinations[tr:tr+ts]:
                    worlds_dev.append(comb)
                for comb in combinations[tr+ts:]:
                    worlds_test.append(comb)

            print(len(worlds_train), len(worlds_test), len(worlds_dev))

    with open(str(n_goals)+'_train_set' + '.json', 'w') as fo:
        json_object = json.dumps(worlds_train, indent = 4)
        fo.write(json_object)
        fo.write('\n')
    with open(str(n_goals)+'dev_set' + '.json', 'w') as fo:
        json_object = json.dumps(worlds_dev, indent = 4)
        fo.write(json_object)
        fo.write('\n')
    with open(str(n_goals)+'_goals_test_seen' + '.json', 'w') as fo:
        json_object = json.dumps(worlds_test, indent = 4)
        fo.write(json_object)
        fo.write('\n')

    '''- generate unseen environments'''

    worlds_unseen = []
    for env in envs:
        with open(str(sys.argv[1]) + '/' + env) as f:
            data = json.load(f)

            train = 0.8 * len(data)
            test = 0.2 * len(data)

            for inst in data[int(train):]:
                obstacles = inst['obstacles']
                shape = inst['shape']

                print(obstacles)
                combinations = generate_worlds(obstacles, shape[0], n_goals, trials=30)

                tr = int(0.8 * len(combinations))
                dv = int(0.1 * len(combinations))
                ts = int(0.1 * len(combinations))
                
                print(tr, dv, ts)
                print(len(combinations))
                for comb in combinations:
                    worlds_unseen.append(comb)

            print(len(worlds_unseen))

    with open(str(n_goals)+'goals_unseen' + '.json', 'w') as fo:
        json_object = json.dumps(worlds_unseen, indent = 4)
        fo.write(json_object)
        fo.write('\n')

if __name__ == "__main__":
    main()