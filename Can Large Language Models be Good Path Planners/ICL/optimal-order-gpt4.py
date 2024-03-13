import openai
import os 
import json
import time
import tiktoken

openai.organization = "ORGANIZATION-KEY"
key = "API-KEY"
openai.api_key = key
openai.Model.list()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


eng = []
acts = []

eng = json.load(open('test-sets/updated/ICL_test_set_multigoal_updated.json'))
with open('prompts/ordering-prompts-optimal.txt', 'r') as f:
    prompt = f.read()

out = []

inp_tokens = 0

inputs = []
for i in range(len(eng)):

    try:

        prompt_=f""" {prompt[:-1]}
###
Task: {eng[i]['nl_description']}
Order: 
        """
        print(prompt_)

        inputs.append({
            'prompt': prompt_,
            'token_count': inp_tokens
        })

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages = [{"role": "user", "content": prompt_}],
            temperature = 0.0,
            max_tokens=250
        )

        print(response)
        test = {
            "english": eng[i]['nl_description'],
            "ground_truth":eng[i]['solution_inspect'],
            "predicted": response['choices'][0]['message']['content'],
            'world': eng[i]['world'],
            'prompt_tokens': response['usage']['prompt_tokens'],
            'gen_tokens': response['usage']['completion_tokens']
        }
          
        out.append(test)
    except Exception as e: 
        print(e)
        time.sleep(60)
        prompt_=f""" {prompt[:-1]}
###
Task: {eng[i]['nl_description']}
Actions: 
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages = [{"role": "user", "content": prompt_}],
            temperature = 0.0,
            max_tokens=250
        )

        print(response)
        test = {
            "english": eng[i]['nl_description'],
            "ground_truth":eng[i]['agent_as_a_point'],
            "predicted": response['choices'][0]['message']['content'],
            'world': eng[i]['world'],
            'prompt_tokens': response['usage']['prompt_tokens'],
            'gen_tokens': response['usage']['completion_tokens']
        }
          
        out.append(test)

with open('outputs/GPT-4-outputs-opt-ordering.json', 'w') as fo:
      json_object = json.dumps(out, indent = 4)
      fo.write(json_object)
      fo.write('\n')
   