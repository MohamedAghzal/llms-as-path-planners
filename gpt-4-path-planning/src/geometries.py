import random

def rectangle(size, n_blocks = 3):

    def all_true(arr):
        val = True
        for v in arr:
            val &= v
        return val 

    def no_overlap(rect, points):
        valid = True
        for pts in points:
            valid = (all_true([(p[0] < pts[0][0] - 1) for p in rect]) 
                      or all_true([(p[0] > pts[1][0] + 1) for p in rect])
                      or all_true([(p[1] > pts[3][1] + 1) for p in rect])
                      or all_true([(p[1] < pts[0][1] - 1) for p in rect])
                    )

        return valid

    '''
        size: size of the grid
        n_blocks: number of rectangles
    '''
    if(n_blocks == 1):
        first_row = random.randint(0, size-3)
        second_row = random.randint(first_row+2, size-1)
        
        first_col = random.randint(0, size-3)
        second_col = random.randint(first_col+2, size-1)
        
        points = [(first_row, first_col), 
                (first_row, second_col),
                (second_row, first_col),
                (second_row, second_col)]

        obstacles = []

        for i in range(first_row, second_row+1):
            for j in range(first_col, second_col+1):
                obstacles.append((i, j))

        return points, sorted(obstacles), {
            'shape': (size, size),
            'obstacles': obstacles,
            'points': points
        }
    else:
        rects = []

        points_seen = []

        obstacles = []

        for _ in range(n_blocks):
            first_row = random.randint(0, size-5)
            second_row = random.randint(first_row+2, size-1)
            
            first_col = random.randint(0, size-5)
            second_col = random.randint(first_col+2, size-1)
            
            points = [(first_row, first_col), 
                    (first_row, second_col),
                    (second_row, first_col),
                    (second_row, second_col)]
            

            for p in points_seen:
                print(p)
            while(not no_overlap(points, points_seen)):
                
                first_row = random.randint(0, size-3)
                second_row = random.randint(first_row+2, size-1)
                
                first_col = random.randint(0, size-3)
                second_col = random.randint(first_col+2, size-1)
                
                points = [(first_row, first_col), 
                        (first_row, second_col),
                        (second_row, first_col),
                        (second_row, second_col)]
                

            points_seen.append(points)

            for i in range(first_row, second_row+1):
                for j in range(first_col, second_col+1):
                    obstacles.append((i, j))

            rects.append([points, sorted(obstacles), {
                'shape': (size, size),
                'obstacles': obstacles,
                'points': points
            }])

        return points_seen, sorted(obstacles), {
                'shape': (size, size),
                'obstacles': obstacles,
                'points': points_seen
            }

def triangle(size, orientation = 'left', triangle_size=None):
    """
    size: Size of the grid.
    triangle_size: The length of the legs of the right-angled triangle. If None, randomly determined.
    """
    if triangle_size is None:
        triangle_size = random.randint(3, size // 2)   # Randomly determine the size, at least 3 for visibility
    
    # Randomly determine the starting point (right angle vertex)
    start_row = random.randint(0, size - triangle_size)
    start_col = random.randint(0, size - triangle_size)

    obstacles = []
    
    points = []
    if orientation == 'left':
        points = [(start_row, start_col), 
                  (start_row, start_col + triangle_size), 
                  (start_row + triangle_size, start_col)]
        
        for i in range(start_row, start_row + triangle_size):
            for j in range(start_col, start_col + triangle_size):
                if i - start_row + j - start_col < triangle_size:
                    obstacles.append((i, j))
                    
    if orientation == 'right':
        points = [(start_row, start_col), 
            (start_row + triangle_size, start_col + triangle_size), 
            (start_row + triangle_size, start_col)]
        
        n_d = 1
        for i in range(start_row, start_row + triangle_size):
            for j in range(start_col, start_col + n_d):
                    obstacles.append((i, j))
            n_d += 1
        
    elif orientation == 'down':
        triangle_size = triangle_size + (1 - triangle_size & 1)
        s = triangle_size
        
        points = [(start_row,  (size - s) // 2), 
                (start_row , (size - s) // 2 + s), 
                (start_row + triangle_size, (size - s) // 2 + s // 2)]
        
        for i in range(start_row, start_row + triangle_size):
            diff = (size - s) // 2
            print('Difference', diff)
            for j in range(diff, diff+s):
                obstacles.append((i, j))
            s = s - 2
            
    elif orientation == 'up':
        triangle_size = triangle_size + (1 - triangle_size & 1)
        s = 1
        
        points = [
                (start_row, (size - s) // 2 + s // 2),
                (start_row + triangle_size,  (size - s) // 2), 
                (start_row + triangle_size, (size - s) // 2 + s),
                ]
        
        for i in range(start_row, start_row + triangle_size):
            diff = (size - s) // 2
            print('Difference', diff)
            for j in range(diff, diff+s):
                obstacles.append((i, j))
            s = s + 2
            
            
    return points, sorted(obstacles), {
        'shape': (size, size),
        'obstacles': obstacles,
        'points': points
    }

def spiral_maze(size):
    
    # Ensure the size is odd to have a single center point
    if size % 2 == 0:
        size += 1

    # Initialize the grid with all paths (0's)
    maze = [[0 for _ in range(size)] for _ in range(size)]

    # Create nested squares
    obstacles = []
    for i in range(0, size // 2, 2):
        # Top horizontal line
        opening = False
        for x in range(i, size - i):
            maze[i][x] = 1
            r = random.random()
            if(x == i): continue
            if(r > 0.7 and opening==False):
                maze[i][x] = 0
                opening = True
            else:
                obstacles.append([i, x])
        # Right vertical line
        for y in range(i, size - i):
            maze[y][size - i - 1] = 1
            r = random.random()
            if(x == i): continue

            if(r > 0.7 and opening==False):
                maze[y][size - i - 1] = 0
                opening = True
            else:
                obstacles.append([y, size - i - 1])
        # Bottom horizontal line
        for x in range(size - i - 1, i, -1):
            maze[size - i - 1][x] = 1
            
            r = random.random()
            if(x == i): continue

            if(r > 0.7 and opening==False):
                maze[size - i - 1][x] = 0
                opening = True
            else:
                obstacles.append([size - i - 1, x])
            
        # Left vertical line
        for y in range(size - i - 1, i, -1):
            maze[y][i] = 1

            r = random.random()
            if(x == i): continue

            if(r > 0.7 and opening==False or (x == i + 1 and opening == False)):
                maze[y][i] = 0
                opening = True
            else:
                obstacles.append([y, i])
            

    for i in range(0, size, 2):
        if([i, i] not in obstacles):
            obstacles.append([i, i])

    return sorted(obstacles), {
        'shape': (size, size),
        'obstacles': obstacles
    }

def random_obstacles(size):
    n_obstacles = random.randint(5, 50)

    obstacles = []
    while len(obstacles) < n_obstacles:
        obst_x = random.randint(0, size)
        obst_y = random.randint(0, size)

        if([obst_x, obst_y] in obstacles):
            continue
        obstacles.append([obst_x, obst_y])

    return sorted(obstacles), {
        'shape': (size, size),
        'obstacles': obstacles
    }

def sample(shape, choice, target):
    '''
        - shape: shape of the grids
        - choice: rectangle, triangle, maze
        - target: number of samples
    '''

    envs = []
    out = []

    j = 0
    t_choices = ['left', 'up', 'right', 'down']
    while(True):
        if(choice == 'rectangle'):
            points, obsts, env = rectangle(shape)
        if(choice == 'triangle'):
            points, obsts, env = triangle(shape, t_choices[j%4])
            j += 1
        elif(choice == 'maze'):
            obsts, env = spiral_maze(shape)
        elif(choice == 'random'):
            obsts, env = random_obstacles(shape)

        if(obsts in envs):
            continue
        
        envs.append(obsts)
        out.append(env)
        print("Generating environment #", len(envs))
        if(len(envs) == target):
            break
        
    return out
    
    


