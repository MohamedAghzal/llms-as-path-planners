import json
import openai
import time

openai.organization = "org"
key = "api-key"

openai.api_key = key
openai.Model.list()

def world_description(nl_description):
    return nl_description.split('You are at')[0]

def execute(actions, start, goal, world):

    acts = ['up', 'down', 'left', 'right']

    nx = len(world)
    ny = len(world[0])

    sequence = actions.replace('.','').split(' ')

    if(start == (-1, -1)): #First run
        for k in range(len(world)):
            for p in range(len(world[0])):
                print(world[k][p], end = ' ')
                if(world[k][p] == 2):
                    pos = (k, p)
    else:
        pos = start

    nl_return = ''
    i = 0
    for action in sequence:
        x = pos[0]
        y = pos[1]

        orig = (x, y)

        print(action)
        if(action == 'left'):
            pos = (x, y-1)
        if(action == 'right'):
            pos = (x, y+1)
        if(action == 'down'):
            pos = (x+1, y)
        if(action == 'up'):
            pos = (x-1, y)

        ret_pos = ()
        ret_val = 0
        if(pos[0] < 0 or pos[1] < 0 or pos[0] >= nx or pos[1] >= ny):
            nl_return += f'After {i} steps, I am at ({orig[0]},{orig[1]}). Performing the next action would lead me outside the grid.'
            ret_val = -1 #outside the grid
            ret_pos = (orig[0],orig[1])
            break
        elif(world[pos[0]][pos[1]] == 1):
            nl_return += f'After {i} steps, I am at ({orig[0]},{orig[1]}). If I perform the next step I will run into the obstacle at ({pos[0]},{pos[1]}).'
            ret_val = 2 #ran into an obstacle
            ret_pos = (orig[0],orig[1])
            break
        i += 1
    
    if(ret_val == 0):
        if(pos == goal):
            nl_return += f'Performing the action sequence leads to ({pos[0]},{pos[1]}). The task has been solved'
            ret_val = 1 #Task solved
            ret_pos = (pos[0],pos[1])
        else: 
            nl_return += f'Performing the action sequence leads to ({pos[0]},{pos[1]}). The task has not been solved yet.'
            ret_val = 0 #Task not solved
            ret_pos = (pos[0],pos[1])
    
    return ret_val, nl_return, ret_pos


def get_next(source, dest, world_desc, world):
    with open('prompts/CoT-SG.txt', 'r') as f:
        prompt = f.read()
    
    prompt_=f""" {prompt[:-1]}
###
Task: {f'{world_desc} Go from ({source[0]}, {source[1]}) to ({dest[0]}, {dest[1]}).'}
Actions: 
     """
    try:
        response = openai.ChatCompletion.create(
                model="gpt-4",
                messages = [{"role": "user", "content": prompt_}],
                temperature = 0.0,
                max_tokens=250
        )

        acts = response['choices'][0]['message']['content'].replace('Therefore ', 'Therefore, ')
        
        if(len(acts.split('Therefore, my action sequence is: ')) == 2):
            thought = acts.split('Therefore, my action sequence is: ')[0]
            actions = acts.split('Therefore, my action sequence is: ')[1]
        else:
            thought = acts
            actions = 'No action'

        execution = execute(actions, source, dest, world)
    except:
        time.sleep(60)
        response = openai.ChatCompletion.create(
                model="gpt-4",
                messages = [{"role": "user", "content": prompt_}],
                temperature = 0.0,
                max_tokens=250
        )

        acts = response['choices'][0]['message']['content'].replace('Therefore ', 'Therefore, ')
        
        if(len(acts.split('Therefore, my action sequence is: ')) == 2):
            thought = acts.split('Therefore, my action sequence is: ')[0]
            actions = acts.split('Therefore, my action sequence is: ')[1]
        else:
            thought = acts
            actions = 'No action'

        execution = execute(actions, source, dest, world)

    trials = 1

    nl = f'{world_desc} Go from ({source[0]}, {source[1]}) to ({dest[0]}, {dest[1]}).'
    
    with open('prompts/react-prompt.txt', 'r') as f:
        prompt = f.read()

    history = [
    f"""{prompt[:-1]}
### 
Task: {nl}
Thought 1: {thought}
Act 1: {actions}
Obs 1: {execution[1]}
    """
    ]

    agent = execution[2]
    position = agent
    outputs = []
    if(execution[0] != 1):
        while(trials < 3):
            try:
                messages = []
                for message in history:
                    messages.append({
                        "role":"user",
                        "content":message 
                    })
                
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages = messages,
                    temperature = 0.0,
                    max_tokens=1000,
                    stop=f'Obs {trials+1}'
                )

                parse = response['choices'][0]['message']['content'].split('\n')
                thought = parse[0]
                resp = parse[1].replace('.','').split(':')[1]

                execution = execute(resp, agent, dest, world)
                agent = execution[2]
                print(execution[1])
                history.append(
                f"""
Thought {trials + 1}: {thought.replace(f'Thought {trials + 1}: ', '')}
Act {trials + 1}: {resp}
Obs {trials + 1}: {execution[1]}
                """
                )

                position = execution[2]

                if(execution[0] == 1 or sample['solution_inspect'] == '' and 'No action' in resp):
                    break
                
                trials += 1
            except:

                time.sleep(60)
                messages = []
                for message in history:
                    messages.append({
                        "role":"user",
                        "content":message 
                    })
                
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages = messages,
                    temperature = 0.0,
                    max_tokens=1000,
                    stop=f'Obs {trials+1}'
                )

                parse = response['choices'][0]['message']['content'].split('\n')
                thought = parse[0]
                resp = parse[1].replace('.','').split(':')[1]

                execution = execute(resp, agent, dest, world)
                agent = execution[2]
                print(execution[1])
                history.append(
                f"""
Thought {trials + 1}: {thought.replace(f'Thought {trials + 1}: ', '')}
Act {trials + 1}: {resp}
Obs {trials + 1}: {execution[1]}
                """
                )

                position = execution[2]

                if(execution[0] == 1):
                    break
                
                trials += 1

    return position, execution[0], history

def extract_goals(nl_description):
    
    goals = []

    positions = nl_description.split('.')[4].split('p')

    for e in positions:
        
        if(e == ' '):
            continue
            
        sp = e.split(' is located at ')
        
        goal_num = sp[0].replace(' ', '')
        location = sp[1].replace(' ', '').replace('),', '').replace(')', '').replace('(','').replace('and','').split(',')
        
        goals.append((int(location[0]), int(location[1])))
        
    print(goals)
    
    return goals

def convert_to_coordinates(nl_description, ordering):
    # Parse nl_description to extract the coordinates. 
    # Return the coordinates of the locations to be visited

    goals = extract_goals(nl_description)

    coords = []
    for i in range(len(ordering)):
        coords.append(goals[ordering[i]])

    return coords

def get_ordering(sample):

    with open('prompts/ordering_prompts.txt') as f:
        prompt = f.read()


    try:
        prompt_=f""" {prompt[:-1]}
###    
Task: f{sample['nl_description']}
Order: 
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",            
            messages = [{"role": "user", "content": prompt_}],
            temperature = 0.0,
            max_tokens=250,
        )

        order = response['choices'][0]['message']['content']

        print(order)

        ordering = [int(i) for i in order.replace('p','').replace(' ', '').split(',')]

        coords = convert_to_coordinates(sample['nl_description'], ordering)
    except:
        time.sleep(60)

        prompt_=f""" {prompt[:-1]}
###
Task: f{sample['nl_description']}
Order: 
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",            
            messages = [{"role": "user", "content": prompt_}],
            temperature = 0.0,
            max_tokens=250,
        )

        order = response['choices'][0]['message']['content']

        print(order)

        ordering = [int(i) for i in order.replace('p','').replace(' ', '').split(',')]

        coords = convert_to_coordinates(sample['nl_description'], ordering)


    return coords, ordering

def check_constraint(ordering, constraint):

    if('before'in constraint):
        first = constraint.split('before')[0]
        second = constraint.split('before')[1]

        before = []
        after = []

        for i in range(len(first)):
            if (first[i] == 'p'):
                before.append(int(first[i+1]))

        for i in range(len(second)):
            if (second[i] == 'p'):
                after.append(int(second[i+1]))

        visited = []

        for pos in ordering:
            if(pos in after):
                for j in before:
                    if(j not in visited):
                        return False
            
            visited.append(pos)

        if(len(ordering) != len(before) + len(after)):
            return False

    return True

def hierarchy(sample):

    coords, order = get_ordering(sample)
    
    constr = ''
    if('before' in sample['nl_description']):
        constr = sample['nl_description'].split('.')[-1]

    contr = check_constraint(order, constr)

    visited = []

    histories = []

    world = sample['world']

    pos = (-1, -1)
    for i in range(len(world)):
        for j in range(len(world[i])):
            if(world[i][j] == 2):
                pos = (i, j)

    visited.append(pos)
    coords = [pos] + coords
    for i in range(len(coords) - 1):

        pos, success, history = get_next(pos, coords[i+1], world_description(sample['nl_description']), sample['world'])
        if(success == 1):
            visited.append(pos)
        
        histories.append((success, history))


    goals_visited = True
    for i in range(len(coords)):
        if(coords[i] not in visited):
            goals_visited = False

    return goals_visited, contr, visited, pos, histories, order

f = open('test-sets/updated/ICL_test_set_multigoal_updated.json')

eng = json.load(f)

output = []
i = 0
for sample in eng[314:]:

    goals_visited, const, visited, pos, hist, order = hierarchy(sample)
    i += 1
    output.append({
        'id': i,
        'world': sample['world'],
        'nl_description': sample['nl_description'],
        'history': hist,
        'ground_truth': sample['solution_inspect'],
        'constraint_satisfied': const,
        'all_goals_visited': goals_visited,
        'final_pos': pos,
        'visited_list': visited,
        'predicted_order': order
    })

    with open('outputs/hierarchy-out-4.json', 'w') as f:
        obj = json.dumps(output, indent=4)
        f.write(obj)
        f.write('\n')


    
    