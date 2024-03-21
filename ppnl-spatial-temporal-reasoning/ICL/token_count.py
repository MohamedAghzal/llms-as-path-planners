import tiktoken
import json

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

out = num_tokens_from_string("tiktoken is great!", "cl100k_base")
print(out)

eng = []
acts = []

eng = json.load(open('ICL_test_set.json'))

out = []

inp_tokens = 0

inputs = []

prompt = ''

print(prompt)
with open('CoT-SG.txt') as f:
    prompt = f.read()
    
for i in range(61, 62):
        print(i)
        prompt_=f"""{prompt}
Task: {eng[i]['nl_description']}
Actions: 
        """
        
        print(prompt_)
        inp_tokens += num_tokens_from_string(prompt_, "cl100k_base")

        inputs.append({
            'prompt': prompt_,
            'token_count': num_tokens_from_string(prompt_, "cl100k_base")
        })

outputs = []

'''f = open('GPT-4-outputs-test.json')
out = json.load(f)

gen = 0
for item in out:
    gen += num_tokens_from_string(item['Predicted'], "cl100k_base")
    outputs.append({
        'output': item['Predicted'],
        'token_count': num_tokens_from_string(item['Predicted'], "cl100k_base")
    })'''

with open('prompts_token_count.json', 'w') as fo:
    json_object = json.dumps(inputs)
    fo.write(json_object)
    fo.write('\n')

print(inp_tokens)
