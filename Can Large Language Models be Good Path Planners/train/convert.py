import json

f = open('../multi-goal/multigoal_train_none_order0.json')

data = json.load(f)

spider = []
for item in data:
    spider.append({
        'question': item['nl_description'],
        'target': item['solution_inspect']
    })

with open('T5/transformers_cache/multigoal_train_updated.json', 'w') as fo:
        json_object = json.dumps(spider, indent = 4)
        fo.write(json_object)
        fo.write('\n')

with open('BART/transformers_cache/multigoal_train_updated.json', 'w') as fo:
        json_object = json.dumps(spider, indent = 4)
        fo.write(json_object)
        fo.write('\n')