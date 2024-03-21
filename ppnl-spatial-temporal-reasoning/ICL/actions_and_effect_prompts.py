import json

def count_obstacles(world):
    cnt = 0
    for i in range(len(world)):
        for j in range(len(world[i])):
            if(world[i][j]==1):
                cnt += 1
    return cnt

def actions_and_effects(actions, path):
    print(actions.split(' '))
    print(path)
    prompt = ''
    if('Goal not reachable' in path):
        return 'Goal not reachable'
    steps = actions.split(' ')
    for i in range(len(steps)):
        if(i + 1 < len(steps)):
            prompt += 'Go ' + steps[i] + '. You are now at ' + f'({path[i + 1][0]},{path[i + 1][1]})' +  '. '

    prompt += f'Hence, the action sequence is: {actions}'    

    return prompt

out = []

nl_train = []
actions_train = []
worlds = []

nl_dev = []
actions_dev = []
paths = []




with open('../single_goal/6x6worlds/train_set.json') as f:
    data = json.load(f)
        
    for instance in data:
        nl_train.append(instance['nl_description'])
        actions_train.append(instance['agent_as_a_point'])
        worlds.append(instance['world'])
        paths.append(instance['solution_coordinates'])


prompt = f'Provide a sequence of actions to navigate a world to reach a goal similarly to the examples below. (0,0) is located in the upper-left corner and (M, N) lies in the M row and N column.\n'

n = 10

per_obst = n / 5

few_shot = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0
}

seen_envs = []

for i in range(len(nl_train)):
    n_obsts = count_obstacles(worlds[i])
    env = worlds[i]

    for k in range(len(env)):
        for d in range(len(env[k])):
            if(env[k][d] == 2 or env[k][d] == 3):
                env[k][d] = 0
    
    if(few_shot[n_obsts] < per_obst and (env not in seen_envs)):
        prompt += f'###\n'
        prompt += f'Task: {nl_train[i]}\n'

        steps = actions_and_effects(actions_train[i], paths[i])
        
        prompt += f'Actions: {steps}\n'
        seen_envs.append(env)
        few_shot[n_obsts] += 1
    
    total_examples = 0
    for p in few_shot:
        total_examples += few_shot[p]
    
    if(total_examples >= n):
        break

with open(f'action-effect-prompts-{n}.txt', 'w') as f:
    f.write(prompt)

    


    
