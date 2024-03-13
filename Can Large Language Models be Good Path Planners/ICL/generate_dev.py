import json

text = ''
with open('few-shot-prompts.txt') as f:
    for line in f:
        text += line

print(text)   
f = open('../single_goal/6x6worlds/dev_set.json')
data = json.load(f)

examples = text.split('###')

fs = []
for example in examples:
    fs.append(example.split('\n'))

ex = []
seen = []
unseen = []

for example in fs:
    for sub in example:
        if('Task' in sub):
            ex.append(sub.split('Task: ')[1].split('Go from')[0])

for item in data:
    nl = item['nl_description']
    path = item['agent_as_a_point']
    world = item['world']

    if(nl.split('Go from')[0] in ex):
        seen.append({
            'nl_description': nl,
            'agent_as_a_point': path,
            'world': world
        })
    else:
        unseen.append({
            'nl_description': nl,
            'agent_as_a_point': path,
            'world': world
        })

with open('dev_seen.json', 'w') as fo:
    json_object = json.dumps(seen, indent = 4)
    fo.write(json_object)
    fo.write('\n')

with open('dev_unseen.json', 'w') as fo:
    json_object = json.dumps(unseen, indent = 4)
    fo.write(json_object)
    fo.write('\n')

print(len(seen), len(unseen))



