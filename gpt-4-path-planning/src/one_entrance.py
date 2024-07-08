import random

def one_opening_per_row(size, even_or_odd): 
    #even_or_odd: where obstacles are to be placed, 0 if at even rows, 1 if at odd rows
    obstacles = []
    for i in range(size):
        entrance = (i % 2 == even_or_odd)
        value = [-1]
        if(entrance == True):
            candidates = [k for k in range(0, size)]
            value = random.sample(candidates, 1)

        for j in range(size):
            if(entrance and j != value[0]):
                obstacles.append((i, j))

    return sorted(obstacles), {
        'shape': (size, size),
        'obstacles': obstacles
    }


def one_opening_per_column(size, even_or_odd): 
    #even_or_odd: where obstacles are to be placed, 0 if at even columns, 1 if at odd columns
    obstacles = []
    for i in range(size):
        entrance = (i % 2 == even_or_odd)
        value = [-1]
        if(entrance == True):
            candidates = [k for k in range(0, size)]
            value = random.sample(candidates, 1) #double check

        for j in range(size):
            if(entrance and j != value[0]):
                obstacles.append((j, i))

    return sorted(obstacles), {
        'shape': (size, size),
        'obstacles': obstacles
    }

def sample(shape, choice, target):
    '''
        - shape: shape of the grids
        - choice: row, column, diagonal
        - target: number of samples
    '''

    envs = []
    out = []

    print(type(shape))
    while(True):
        if(choice == 'column'):
            obsts, env = one_opening_per_column(shape, 1)
        if(choice == 'row'):
            obsts, env = one_opening_per_row(shape, 1)
        
        print(obsts)
        if(obsts in envs):
            continue
        
        envs.append(obsts)
        out.append(env)
        print("Generating environment #", len(envs))
        if(len(envs) == target):
            break
        
    return out
