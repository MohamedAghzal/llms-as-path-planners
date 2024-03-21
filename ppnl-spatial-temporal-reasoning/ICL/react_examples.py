import openai
import os 
import json
import time
import tiktoken
import random

openai.organization = "org"
key = "api-key"
openai.api_key = key
openai.Model.list()

tr_file = open('ReAct-5examples.json')
eng = json.load(tr_file)


with open('prompts/CoT-SG.txt', 'r') as f:
    prompt = f.read()
print(eng[0])
for i in range(len(eng)):
    print(eng[i]['nl_description'])
    if(eng[i]['nl_description'] in prompt):
        print('wohoo')

out = []

inp_tokens = 0

inputs = []
for i in range(len(eng)):

    try:

        prompt_=f""" {prompt[:-1]}
###
Task: {eng[i]['nl_description']}
Actions: 
        """
        print(prompt_)

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

with open('ReAct-CoT-examples-5examples.json', 'w') as fo:
      json_object = json.dumps(out, indent = 4)
      fo.write(json_object)
      fo.write('\n')
