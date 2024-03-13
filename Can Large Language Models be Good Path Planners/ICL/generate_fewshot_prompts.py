import json

out = []

nl_train = []
actions_train = []
worlds = []

nl_dev = []
actions_dev = []


def count_obstacles(world):
    cnt = 0
    for i in range(len(world)):
        for j in range(len(world[i])):
            if(world[i][j]==1):
                cnt += 1
    return cnt

def count_goals(world):
    cnt = 0
    for i in range(len(world)):
        for j in range(len(world[i])):
            if(world[i][j]==3):
                cnt += 1
    return cnt

with open('../single_goal/6x6worlds/train_set.json') as f:
    data = json.load(f)
        
    for instance in data:
        nl_train.append(instance['nl_description'])
        actions_train.append(instance['agent_as_a_point'])
        worlds.append(instance['world'])

with open('../single_goal/6x6worlds/dev_set.json') as f:
    data = json.load(f)
        
    for instance in data:
        nl_dev.append(instance['nl_description'])
        actions_dev.append(instance['agent_as_a_point'])

prompt = f'Provide a sequence of actions to navigate a world to reach a goal similarly to the examples below. (0,0) is located in the upper-left corner and (M, N) lies in the M row and N column.\n'

n = 15

per_obst = n / 5

few_shot = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0
}


few_shot_goals = {
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
}

seen_envs = []
n_goals = count_goals(worlds[0])
if(n_goals > 1):

    per_goal = n / 5

    for i in range(len(nl_train)):
        n_obsts = count_obstacles(worlds[i])
        env = worlds[i]
        n_goals = count_goals(worlds[i])
        for k in range(len(env)):
            for d in range(len(env[k])):
                if(env[k][d] == 2 or env[k][d] == 3):
                    env[k][d] = 0
        
        print(nl_train[i])
        if(few_shot[n_obsts] < per_obst and few_shot_goals[n_goals] < per_goal and (nl_train[i] not in seen_envs)):
            prompt += f'###\n'
            prompt += f'Task: {nl_train[i]}\n'
            prompt += f'Actions: {actions_train[i]}\n'
            seen_envs.append(nl_train[i])
            few_shot[n_obsts] += 1
            few_shot_goals[n_goals] += 1
        
        total_examples = 0
        for p in few_shot:
            total_examples += few_shot[p]
        
        if(total_examples >= n):
            break

else:
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
            prompt += f'Actions: {actions_train[i]}\n'
            seen_envs.append(env)
            few_shot[n_obsts] += 1
        
        total_examples = 0
        for p in few_shot:
            total_examples += few_shot[p]
        
        if(total_examples >= n):
            break

with open('few-shot-prompts-single_goal_15.txt', 'w') as f:
    f.write(prompt)

    


    
