import openai
import os 
import json
import time
import tiktoken

openai.organization = "org-hNldcAS0NRKQWySC1hU3BId8"
key = "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"
openai.api_key = key
openai.Model.list()

def execute(actions, start, world, nx=6, ny=6):

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
            ret_val = -1 #outside the 
            ret_pos = (orig[0],orig[1])
            break
        elif(world[pos[0]][pos[1]] == 1):
            nl_return += f'After {i} steps, I am at ({orig[0]},{orig[1]}). If I perform the next step I will run into the obstacle at ({pos[0]},{pos[1]}).'
            ret_val = 2 #ran into an obstacle
            ret_pos = (orig[0],orig[1])
            break
        i += 1    
    
    if(ret_val == 0):
        if(world[pos[0]][pos[1]] == 3):
            nl_return += f'Performing the action sequence leads to ({pos[0]},{pos[1]}). The task has been solved'
            ret_val = 1 #Task solved
            ret_pos = (pos[0],pos[1])
        else: 
            nl_return += f'Performing the action sequence leads to ({pos[0]},{pos[1]}). The task has not been solved yet.'
            ret_val = 0 #Task not solved
            ret_pos = (pos[0],pos[1])
    
    return ret_val, nl_return, ret_pos

def reAct(test_case, prompt, execute):
    
    nl = test_case['nl_description']

    if(len(test_case['CoT'].split('Therefore, my action sequence is: ')) == 2):
        thought = test_case['CoT'].split('Therefore, my action sequence is: ')[0]
        actions = test_case['CoT'].split(' Therefore, my action sequence is: ')[1]
    else:
        thought = test_case['CoT']
        actions = 'No action'

    execution = execute(actions, (-1,-1), test_case['world'])

    trials = 1

    
    history = [
    f"""
{prompt}
###
Task: {nl}
Thought 1: {thought}
Act 1: {actions}
Obs 1: {execution[1]}
    """
    ]

    agent = execution[2]

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

            test = {
                "english": eng[i]['nl_description'],
                "ground_truth":eng[i]['agent_as_a_point'],
                "predicted": response['choices'][0]['message']['content'],
                'world': eng[i]['world'],
                'prompt_tokens': response['usage']['prompt_tokens'],
                'gen_tokens': response['usage']['completion_tokens']
            }
            
            print(test)
            parse = response['choices'][0]['message']['content'].split('\n')
            thought = parse[0]
            resp = parse[1].replace('.','').split(':')[1]

            execution = execute(resp, agent, test_case['world'])
            agent = execution[2]
            print(execution[1])
            exect = execution[1]

            if(exect == 'No action'):
                if('Goal not reachable' in test_case['ground_truth']):
                    exect = 'No action is to be performed. The goal is not reachable. The task has been solved.'

            history.append(
            f"""
Thought {trials + 1}: {thought.replace(f'Thought {trials + 1}: ', '')}
Act {trials + 1}: {resp}
Obs {trials + 1}: {exect}
            """
            )

            if(execution[0] == 1):
                break
            
            trials += 1
        except:

            time.sleep(20)
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

            test = {
                "english": eng[i]['nl_description'],
                "ground_truth":eng[i]['agent_as_a_point'],
                "predicted": response['choices'][0]['message']['content'],
                'world': eng[i]['world'],
                'prompt_tokens': response['usage']['prompt_tokens'],
                'gen_tokens': response['usage']['completion_tokens']
            }
            
            print(test)
            parse = response['choices'][0]['message']['content'].split('\n')
            thought = parse[0]
            resp = parse[1].replace('.','').split(':')[1]

            execution = execute(resp, agent, test_case['world'])
            agent = execution[2]
            print(execution[1])
            exect = execution[1]

            if(exect == 'No action'):
                if('Goal not reachable' in test_case['ground_truth']):
                    exect = 'No action is to be performed. The goal is not reachable. The task has been solved.'

            history.append(
            f"""
Thought {trials + 1}: {thought.replace(f'Thought {trials + 1}: ', '')}
Act {trials + 1}: {resp}
Obs {trials + 1}: {exect}
            """
            )

            if(execution[0] == 1):
                break
            
            trials += 1
            
    out_str = ''    
    for x in history:
        out_str += x + '\n'

    return history, out_str

eng = json.load(open('test-sets/updated/ReAct_moreobsts_fix_examples.json'))
with open('prompts/react-prompt.txt', 'r') as f:
    prompt = f.read()

outputs = json.load(open('outputs/GPT-4-outputs-ReAct-moreobsts.json'))
for i in range(len(eng[42:])):
    history, out_str = reAct(eng[i], prompt, execute)
    outputs.append({
        'conversation': history,
        'output_string': out_str
    }) 

    with open('outputs/GPT-4-outputs-ReAct-moreobsts-up.json', 'w') as f:
        obj = json.dumps(outputs, indent=4)
        f.write(obj)