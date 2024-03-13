import numpy as np 
import json
import os 
import sys 
import heapq
import gurobipy as gp
from gurobipy import GRB

def generate_nl(x, y, obstacles, goals, initial_loc, constraint='None'):

    '''
        Possible Constraints:
            arithm even
            arithm odd
            arithm prime
            arithm divs {x}
            {x} before {y}
    '''

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
            english += f'p{i} is located at ({goals[i][0]}, {goals[i][1]})'
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

def a_star(grid, start, goal):
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(position):
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

    visited = set()
    heap = []
    heapq.heappush(heap, (0, start, []))
    while heap:
        cost, current, path = heapq.heappop(heap)

        if current == goal:
            return path + [current]

        if current in visited:
            continue

        visited.add(current)

        for action in actions:
            neighbor = (current[0] + action[0], current[1] + action[1])

            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor] != 1:
                    new_cost = cost + 1
                    new_path = path + [current]
                    heapq.heappush(heap, (new_cost + heuristic(neighbor), neighbor, new_path))

    return 'Goal not reachable'

def a_star_value(grid, start, goal):
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def heuristic(position):
            return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

        visited = set()
        heap = []
        heapq.heappush(heap, (0, start, []))

        while heap:
            cost, current, path = heapq.heappop(heap)

            current = tuple(current)
            if current == goal:
                return len(path + [current])

            if current in visited:
                continue

            visited.add(current)

            for action in actions:
                neighbor = (current[0] + action[0], current[1] + action[1])

                if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor] != 1:
                        new_cost = cost + 1
                        new_path = path + [current]
                        heapq.heappush(heap, (new_cost + heuristic(neighbor), neighbor, new_path))

        return 100000

def solution_point(path):

    if(path == 'Goal not reachable'):
        return path

    directions = ''
    
    for i in range(len(path) - 1):
        curr = path[i]
        nxt = path[i + 1]

        if(curr[0] + 1 == nxt[0]):
            directions += 'down '
        elif (curr[0] - 1 == nxt[0]):
            directions += 'up '
        elif (curr[1] + 1 == nxt[1]):
            directions += 'right '
        elif (curr[1] - 1 == nxt[1]):
            directions += 'left '

    return directions

def solution_direction(path):

    if(path == 'Goal not reachable'):
        return path

    path = path.split(' ')

    directions = ''

    orientation = 'south'
    
    for i in range(len(path)):

        if(path[i] == 'right'):
            if(orientation == 'south'):
                directions += 'turn left move forward '
                orientation = 'east'
            elif (orientation == 'north'):
                directions += 'turn right move forward '
                orientation = 'east'
            elif (orientation == 'east'):
                directions += 'move forward '
            elif (orientation == 'west'):
                directions += 'turn right turn right move forward '
                orientation = 'east'
        elif (path[i] == 'left'):
            if(orientation == 'south'):
                directions += 'turn right move forward '
                orientation = 'west'
            elif (orientation == 'north'):
                directions += 'turn left move forward '
                orientation = 'west'
            elif (orientation == 'east'):
                directions += 'turn left turn left move forward '
                orientation = 'west'
            elif (orientation == 'west'):
                directions += 'move forward '
        elif (path[i] == 'down'):
            if(orientation == 'south'):
                directions += 'move forward '
                orientation = 'south'
            elif (orientation == 'north'):
                directions += 'turn left turn left move forward '
                orientation = 'south'
            elif (orientation == 'east'):
                directions += 'turn right move forward '
                orientation = 'south'
            elif (orientation == 'west'):
                directions += 'turn left move forward '
                orientation = 'south'
        elif (path[i] == 'up'):
            if(orientation == 'south'):
                directions += 'turn right turn right move forward '
                orientation = 'north'
            elif (orientation == 'north'):
                directions += 'move forward '
                orientation = 'north'
            elif (orientation == 'east'):
                directions += 'turn left move forward '
                orientation = 'north'
            elif (orientation == 'west'):
                directions += 'turn right move forward '
                orientation = 'south'

    return directions

def tsp_solver(grid, start, start_with='None'):

    '''
        Possible Constraints:
            even
            odd
            prime
            divs {x}
            {x} before {y}
    '''


    cities = []
    
    print(start)
    
    cities.append(start)

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if((i, j) == start):
                continue
            if(grid[i][j] == 3):
                cities.append((i, j))

    n_goals = len(set(cities))

    print(n_goals)
    distances = []
    for i in range(len(cities)):
        dists_i = []
        for j in range(len(cities)):
            dist = 0
            if(i != j):
                dist = a_star_value(grid, cities[i], cities[j])
                if(dist == 100000):
                    return 'Goal not reachable', []
            dists_i.append(dist)
        distances.append(dists_i)

    n = len(distances)  # Number of cities


    # Create a new Gurobi model
    model = gp.Model("TSP")

    # Create decision variables
    visit = model.addVars(n, n, vtype=GRB.BINARY, name="visit")
    u = model.addVars(n, vtype=GRB.INTEGER, name="order")
    #position = model.addVars(n, vtype=GRB.INTEGER, name="position")

    # Set up the objective function
    model.setObjective(sum(distances[i][j] * visit[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

    # Add constraints
    model.addConstrs(visit.sum(i, '*') == 1 for i in range(n) if (i != j))
    model.addConstrs(visit.sum('*', i) == 1 for i in range(n) if (i != j))
    model.addConstrs(visit[i, i] == 0 for i in range(n))

    for i in range(len(cities)):
        model.addConstr(gp.quicksum(visit[i, j] for j in range(len(cities)) if i != j) == 1, name=f'visit_once_{i}')
        model.addConstr(gp.quicksum(visit[j, i] for j in range(len(cities)) if i != j) == 1, name=f'leave_once_{i}')
    #model.addConstr(sum(distances[i][j] for i in range(n) for j in range(n)) <= 10000)

    n = len(cities)
    for i in range(0, len(cities)):
        for j in range(0, len(cities)):
            if i != j:
                model.addConstr(visit[i, j] + visit[j, i] <= 1, name=f'subtour_elimination_{i}_{j}')
                
                if(i >= 1 and j >= 1):
                    model.addConstr(u[i] - u[j] + n * visit[i, j] <= n - 1, name=f'elim_{i}_{j}')

    num_nodes = len(cities)
    even_nodes = range(0, num_nodes, 2) 
    odd_nodes = range(1, num_nodes, 2) 
    for i in even_nodes:
        for j in odd_nodes:
            if(start_with=='odd'):
                model.addConstr(u[i] + 1 <= u[j], f"parity_{i}_{j}")
            elif (start_with=='even'): 
                model.addConstr(u[j] + 1 <= u[i], f"parity_{i}_{j}")


    if(start_with=='prime'):
        prime = [2, 3, 5, 7, 11, 13, 17, 19]
        for i in prime:
            for j in range(20):
                if(j not in prime):
                    model.addConstr(u[i] + 1 <= u[j], f"primes_{i}_{j}")
    
    if('divisors' in start_with):
        plan = start_with.split(' ')
        targ = int(plan[0])
        divs = []
        for i in range(1, num_nodes):
            if(targ % i == 0):
                divs.append(i)
        
        for i in range(num_nodes):
            if(i in divs): continue
            for j in range(divs):
                model.addConstr(u[i] + 1 <= u[j], f'divs_{i}_{j}')
    
    if('before' in start_with):
        plan = start_with.split('before')
        first = int(plan[0].replace('p', '').replace(' ',''))
        last = int(plan[2].replace('p', '').replace(' ',''))

        model.addConstr(u[first] + 1 <= u[last], f'before_{first}_{last}')

    # Optimize the model    
    model.optimize()

    print(cities)
    # Extract the solution
    if model.status == GRB.OPTIMAL:
        solution = model.getAttr('x', visit)

        # Print the optimal tour
        print("Optimal tour:")
        tour = {}
        print(solution)
        for i in range(n):
            for j in range(n):
                if solution[i, j] > 0.5:
                    tour[cities[i]] = cities[j]
                    print(cities[i], '-->', cities[j])
                    print(i, cities[i], j, cities[j])
        
        path = []

        visited_gls = []
        print(start, '----->', end=' ')
        for k in list(tour.keys()):
            if(k in path):
                continue
            path.append(k)

            curr = tour[k]
            if(k in cities):
                visited_gls.append(k)
            while(curr != k):
                path.append(curr)

                print(k, '---->', curr)
                if(curr in cities):
                    visited_gls.append(curr)
                if(len(set(visited_gls)) == n_goals):
                    break
                curr = tour[curr]
            

            if(len(set(visited_gls)) == n_goals):
                    break
            
        final_path = []
        plan = []

        print(path)
        for i in range(len(path) - 1):
            now = path[i]
            nxt = path[i + 1]

            sub_path = a_star(grid, now, nxt)
            print('from: ', now, 'to: ', nxt, 'path: ', sub_path)
            for id in sub_path:
                final_path.append(id)
            plan.append((now, nxt, sub_path))
        return final_path, plan
    else:
        return "Goal not reachable", []


def solution_plan(plan):

    path = ''
    for i in range(len(plan)):
        curr = plan[i][0]
        nxt = plan[i][1]

        sub_path = solution_point(plan[i][2])
        path += sub_path + 'inspect '

    return path



def main():
    '''
    CLA:
        directory/setting
    '''


    samples = []
    with open(str(sys.argv[1])) as f:
        data = json.load(f)

        for world in data:
            grid = world['world']
            obstacles = world['obstacles']
            start = world['start']
            goals = world['goals']

            print(grid)

            print(goals)

            nl = generate_nl(len(grid), len(grid[0]), obstacles, goals, start)

            if(len(goals) == 1):
                coordinates = a_star(np.array(grid), (start[0], start[1]), (goals[0][0], goals[0][1]))
                sol_point = solution_point(coordinates)
                sol_direct = solution_direction(sol_point)

                sample = {
                    'world': grid,
                    'nl_description': nl,
                    'solution_coordinates': coordinates,
                    'agent_as_a_point': sol_point,
                    'agent_has_direction': sol_direct 
                }
            else:
                coordinates, steps = tsp_solver(np.array(grid), (start[0], start[1]))
                sol_point = solution_point(coordinates)
                sol_direct = solution_direction(sol_point)
                plan = solution_plan(steps)
            
                '''
                    Add:
                    - constraints
                        Arithmetic: ~3200 (train) from each class for each # of goals
                        Before: Random x before y ~20000?
                '''

                sample = {
                    'world': grid,
                    'nl_description': nl,
                    'solution_coordinates': coordinates,
                    'agent_as_a_point': sol_point,
                    'solution_inspect': plan,
                    'agent_has_direction': sol_direct 
                }


            samples.append(sample)

        with open(str(sys.argv[1]).replace('.json', '') + '_samples.json', 'w') as fo:
                json_object = json.dumps(samples, indent = 4)
                fo.write(json_object)
                fo.write('\n')

    
if __name__ == "__main__":
    main()


    