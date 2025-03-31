import json
import sys

file_to_convert = sys.argv[1]
single_or_multi = sys.argv[2]


if "single" in single_or_multi:
    tg = 'agent_as_a_point'
else:
    tg = 'solution_inspect' 
    
f = open(file_to_convert)

data = json.load(f)

spider = []
for item in data:
    spider.append({
        'question': item['nl_description'],
        'target': item[tg]
    })

with open(f'T5/transformers_cache/{file_to_convert.split("/")[-1]}', 'w') as fo:
        json_object = json.dumps(spider, indent = 4)
        fo.write(json_object)
        fo.write('\n')

with open(f'BART/transformers_cache/{file_to_convert.split("/")[-1]}', 'w') as fo:
        json_object = json.dumps(spider, indent = 4)
        fo.write(json_object)
        fo.write('\n')