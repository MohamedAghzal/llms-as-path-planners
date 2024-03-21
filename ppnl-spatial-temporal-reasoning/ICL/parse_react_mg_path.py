import json

def parse_react_path(history, prompt):
    total_path = []
    for goal in history:
        success = goal[0]
        trials = goal[1]

        actions_performed = []
        for trial in trials:
            lines = trial.replace(prompt, '').split('\n')
            cut = -1
            for l in lines:
                if('Act ' in l):
                    sequence = l.split(':')[1].replace('.','').replace('\n', '').split(' ')
                elif('Obs ' in l):
                    observation = l.split(':')[1]
                    if('executing' in l):
                        pp = l.split(' ')
                        for k in range(len(pp)):
                            if(pp[k] == 'executing'):
                                m = {
                                    'first':1,
                                    'second':2,
                                    'third':3,
                                    'fourth':4,
                                    'fifth':5
                                }
                                while(k < len(pp)):
                                    if(pp[k+1] in m):
                                        cut = m[pp[k+1]]
                                        break
                                    elif(pp[k+1].isnumeric()):
                                        cut = int(pp[k+1])
                                        break
                                    k += 1
                                sequence = sequence[:cut]
                    elif('After' in l):
                        pp = l.split(' ')
                        for k in range(len(pp)):
                            if(pp[k] == 'After'):
                                m = {
                                    'first':1,
                                    'second':2,
                                    'third':3,
                                    'fourth':4,
                                    'fifth':5
                                }
                                while(k < len(pp)):
                                    if(pp[k+1] in m):
                                        cut = m[pp[k+1]]
                                        break
                                    elif(pp[k+1].isnumeric()):
                                        cut = int(pp[k+1])
                                        print('Cutting at', cut)
                                        break
                                    k += 1
                                #sequence = sequence[:cut]
                                       

            actions_performed.append(sequence)

        if(success == 1):
            actions_performed.append(['inspect'])
        total_path.append(actions_performed)

    return total_path

#Get rid of CoT prompt in first trial

f = open('outputs/hierarchy-out-1.json')
out1 = json.load(f)
f = open('outputs/hierarchy-out-2.json')
out2 = json.load(f)
f = open('outputs/hierarchy-out-3.json')
out3 = json.load(f)
f = open('outputs/hierarchy-out-4.json')
out4 = json.load(f)

test_set = json.load(open('test-sets/updated/ICL_test_set_multigoal_updated.json'))
out = []

for k in out1:
    out.append(k)

for k in out2:
    out.append(k)

for k in out3:
    out.append(k)

for k in out4:
    out.append(k)


with open('prompts/CoT-SG.txt') as f:
    prompt = f.read()

output_file = []

def get_actual(path, world):
    correct = ''

    nx = len(world)
    ny = len(world[0])

    for i in range(len(world)):
        for j in range(len(world[i])):
            if(world[i][j] == 2):
                pos = (i, j)

    for p in path:
        print(p)
        ok = True
        for l in p:
            for action in l:
                x = pos[0]
                y = pos[1]
                orig = (x, y)
                if(action == 'left'):
                    pos = (x, y-1)
                if(action == 'right'):
                    pos = (x, y+1)
                if(action == 'down'):
                    pos = (x+1, y)
                if(action == 'up'):
                    pos = (x-1, y)
                if(action == 'inspect'):
                    print(pos, 'inspected')
                if(pos[0] < 0 or pos[1] < 0 or pos[0] >= nx or pos[1] >= ny or world[pos[0]][pos[1]] == 1):
                    pos = (orig[0], orig[1])
                    ok = False
                    break
                correct += action + ' '
    
    return correct


for i in out:

    seq = parse_react_path(i['history'], prompt)
    output_string = ''

    instance = {}
    for p in i.keys():
        instance[p] = i[p]

    instance['predicted_sequence'] = get_actual(seq, instance['world'])
    gt = ''

    for hi in test_set:
        if(hi['nl_description'] == i['nl_description']):
            gt = hi['solution_inspect']
            break
    instance['ground_truth'] = gt
    output_file.append(instance)

with open('outputs/react-mg.json', 'w') as f:
    obj = json.dumps(output_file, indent=4)
    f.write(obj)
