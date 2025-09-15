def generate_code_rectangle(x, y, obstacles, goals, initial_loc, points):

    output = f"""
#The goal is to navigate a {x}x{y} grid to go from the initial location to the goal while avoiding obstacles
 
obstacles = []
goals = {[(goals[0][0], goals[0][1])]}
initial_location = {(initial_loc[0], initial_loc[1])}
"""
    for pts in points:
        output += f"""
for i in range({pts[0][0]}, {pts[3][0]}):
    for j in range({pts[0][1]}, {pts[3][1]}):
        obstacles.append((i, j))
        """
    return output

def generate_code_spiral(size_x, size_y, obstacles, goals, initial_loc):
    size = size_x

    ent = []
    for i in range(0, size_x // 2, 2):
        obstacle_set = set(map(tuple, obstacles))

        for x in range(i, size - i):
            if (i, x) not in obstacle_set and (i - 1, x) not in obstacle_set:
                ent.append((i, x))
            if (size - i - 1, x) not in obstacle_set and (size - i, x) not in obstacle_set:
                ent.append((size - i - 1, x))

        for y in range(i, size - i):
            if (y, i) not in obstacle_set and (y, i - 1) not in obstacle_set:
                ent.append((y, i))
            if (y, size - i - 1) not in obstacle_set and (y, size - i) not in obstacle_set:
                ent.append((y, size - i - 1))

    print(size_x, size_y)
    output = f"""
#The goal is to navigate a {size_x}x{size_y} grid to go from the initial location to all goals while avoiding obstacles
 
obstacles = []
goal = {(goals[0][0], goals[0][1])}
initial_location = {(initial_loc[0], initial_loc[1])}
entrances = {ent}

for i in range(0, {size // 2}, 2):
    # Top horizontal line 
    for x in range(i, {size} - i):
        if((i, x) not in entrances):
            obstacles.append((i, x))
    # Right vertical line
    for y in range(i, {size} - i):
        if((y, {size} - i - 1) not in entrances):
            obstacles.append((y, {size} - i - 1))
    # Bottom horizontal line
    for x in range({size} - i - 1, i, -1):
        if(({size} - i - 1, x) not in entrances):
            obstacles.append(({size} - i - 1, x))     
    # Left vertical line
    for y in range({size} - i - 1, i, -1):
        if((y, i) not in entrances):
            obstacles.append((y, i))
            
    for i in range(0, {size}, 2):
        if((i, i) not in obstacles):
            obstacles.append((i, i))
"""

    return output

def generate_code_alternate_column(x, y, obstacles, goals, initial_loc):
    
    ent = []
    for i in range(y):
        obsts_per_row = []
        for obst in obstacles:
            if(obst[1] == i):
                obsts_per_row.append(obst[0])
        if(i % 2 == 0):
            continue

        for k in range(x):
            if(k not in obsts_per_row):
                ent.append((k, i))
    


    output = f"""
#The goal is to navigate a {x}x{y} grid to go from the initial location to the goal location while avoiding obstacles
 
obstacles = []
goals = {(goals[0][0], goals[0][1])}
initial_location = {(initial_loc[0], initial_loc[1])}
entrances = {ent}

for j in range(1, {y}, 2):
    for i in range({x}):
        if((i, j) not in entrances):
            obstacles.append((i, j))
"""
        
    return output

def generate_code_alternate_row(x, y, obstacles, goals, initial_loc):
    
    ent = []
    for i in range(y):
        obsts_per_row = []
        for obst in obstacles:
            if(obst[0] == i):
                obsts_per_row.append(obst[1])
        if(i % 2 == 0):
            continue
        for k in range(x):
            if(k not in obsts_per_row):
                ent.append((i, k))
    


    output = f"""
#The goal is to navigate a {x}x{y} grid to go from the initial location to all goals while avoiding obstacles
 
obstacles = []
goals = {(goals[0][0], goals[0][1])}
initial_location = {(initial_loc[0], initial_loc[1])}
entrances = {ent}

for i in range(1, {x}, 2):
    for j in range({y}):
        if((i, j) not in entrances):
            obstacles.append((i, j))
"""
        
    return output

def generate_code_triangle(x, y, obstacles, goals, initial_loc):
    
    output = f"""
#The goal is to navigate a {x}x{y} grid to go from the initial location to all goals while avoiding obstacles
 
obstacles = []
goals = {goals}
initial_location = {initial_loc}

"""
    for i in range(x):
        obsts_per_row = []

        for obst in obstacles:
            if(obst[0] == i):
                obsts_per_row.append(obst[1])
        
        if(len(obsts_per_row) == 0):
            continue
        
        vals = sorted(obsts_per_row)

        start, end = vals[0], vals[len(vals) - 1]
        ans = -1
                
        add_on = f"""
for j in range({start},{end}): 
    obstacles.append([{i},j])
"""
        
        output += add_on
        
    return output

def naive_enumeration(x, y, obstacles, goals, initial_loc, constraint='None'):

    english = f'You are in a {x} by {y} world. There are obstacles that you have to avoid at: '

    for i in range(len(obstacles)):
        obstacle = obstacles[i]
        english += f'({obstacle[0]},{obstacle[1]})'
        if(i < len(obstacles) - 2):
                english += ', '
        elif (i == len(obstacles) - 2): 
                english += ' and '
    

    english += '. '
    if(len(goals) == 1):
        english += f'Go from ({initial_loc[0]},{initial_loc[1]}) to ({goals[0][0]},{goals[0][1]})' 
    
    else:
        english += f'You are at ({initial_loc[0]},{initial_loc[1]}).'

        english += ' You have to visit ' 
        for i in range(len(goals)):
            english += f'p{i}'
            if(i < len(goals) - 2):
                english += ', '
            elif (i == len(goals) - 2): 
                english += ' and '
        
        english += '. '

        for i in range(len(goals)):
            english += f'p{i} is located at ({goals[i][0]},{goals[i][1]})'
            if(i < len(goals) - 2):
                english += ', '
            elif (i == len(goals) - 2): 
                english += ' and '
        
        english += '. '
        if('arithm' in constraint):
            plan = constraint.split(' ')

            if(len(plan) == 2):
                english += 'Visit ' + plan[1] + ' numbered locations first.'

            elif(len(plan) == 3):
                english += 'Visit divisors of ' + plan[2] + ' first.' 
    return english

def generate_grid(grid):
    grid_str = ''

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            grid_str += f'{grid[i][j]}'
        grid_str += '\n'

    return grid_str