import json
import sys 

f = open(sys.argv[1])
file = json.load(f)

correct = 0
cnt = 0

n_obs = sys.argv[3]
org = json.load(open(sys.argv[2]))

print(len(file), len(org))
seen = []

'''unique = []

for el in file:
    if(el in unique):
        continue
    unique.append(el)

file = unique'''

for p in range(len(file)):
    i = file[p]
    world = org[p]['world']
    '''if(i['ground_truth'] == ''):
        i['ground_truth'] = 'Goal not reachable'
    if(i['generated'][0] == ''):
        i['generated'][0] = 'Goal not reachable'''
    if('Goal not reachable' in org[p]['agent_as_a_point']):
        fnd = 0
        for a in range(len(world)):
            for b in range(len(world)):
                if(world[a][b] == 1):
                    fnd += 1
        #print(fnd)
        if(fnd < int(n_obs)):
            continue

        if('before' not in i['english']):
            continue

        cnt += 1
        if('Goal not reachable' in i['generated'][0]):
            correct += 1

    

print(correct / cnt, cnt, correct)