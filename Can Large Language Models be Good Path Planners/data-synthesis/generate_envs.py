import random
import sys
import json

def generate_environments(n, min_obstacles, max_obstacles):
    obsts = []

    num_obstacles = random.randint(min_obstacles, max_obstacles)
    while(len(obsts) < num_obstacles):
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)

        if((i, j) in obsts):
            continue
        
        obsts.append((i, j))
    
    return {
        'shape': (n, n),
        'obstacles': obsts
    }

def main():
    '''
    CLA:
        shape of the grids
        number of obstacles
        number of environments
    '''
    shape = int(sys.argv[1])
    nb_obstacles = int(sys.argv[2])
    target = int(sys.argv[3])

    envs = []
    while(True):
        env = generate_environments(shape, nb_obstacles, nb_obstacles)
        if(env in envs):
            continue

        envs.append(env)
        if(len(envs) == target):
            break
    with open('environments' + str(nb_obstacles) + '.json', 'w') as fo:
        json_object = json.dumps(envs, indent = 4)
        fo.write(json_object)
        fo.write('\n')

if __name__ == "__main__":
    main()