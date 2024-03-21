from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from transformers import BertTokenizer, EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder, Seq2SeqTrainer, TrainingArguments
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from evaluate import load
import transformers.data
import torch
import json
import os
import sys

def solution_direction(path):

    if(path == 'Goal not reachable'):
        return path

    path = path.split(' ')

    directions = ''

    orientation = 'south'
    
    for i in range(len(path)):

        if(path[i] == 'right'):
            if(orientation == 'south'):
                directions += 'turn left move forward '
                orientation = 'east'
            elif (orientation == 'north'):
                directions += 'turn right move forward '
                orientation = 'east'
            elif (orientation == 'east'):
                directions += 'move forward '
            elif (orientation == 'west'):
                directions += 'turn right turn right move forward '
                orientation = 'east'
        elif (path[i] == 'left'):
            if(orientation == 'south'):
                directions += 'turn right move forward '
                orientation = 'west'
            elif (orientation == 'north'):
                directions += 'turn left move forward '
                orientation = 'west'
            elif (orientation == 'east'):
                directions += 'turn left turn left move forward '
                orientation = 'west'
            elif (orientation == 'west'):
                directions += 'move forward '
        elif (path[i] == 'down'):
            if(orientation == 'south'):
                directions += 'move forward '
                orientation = 'south'
            elif (orientation == 'north'):
                directions += 'turn left turn left move forward '
                orientation = 'south'
            elif (orientation == 'east'):
                directions += 'turn right move forward '
                orientation = 'south'
            elif (orientation == 'west'):
                directions += 'turn left move forward '
                orientation = 'south'
        elif (path[i] == 'up'):
            if(orientation == 'south'):
                directions += 'turn right turn right move forward '
                orientation = 'north'
            elif (orientation == 'north'):
                directions += 'move forward '
                orientation = 'north'
            elif (orientation == 'east'):
                directions += 'turn left move forward '
                orientation = 'north'
            elif (orientation == 'west'):
                directions += 'turn right move forward '
                orientation = 'north'

    return directions

eng = []
acts = []
locs = [] 
n_obs = []
paths = []

print('Running')
with open(sys.argv[1]) as f:
    data = json.load(f)

    for data_point in data:
        eng.append(data_point['nl_description'])
        acts.append(data_point['solution_inspect'])



print(len(set(eng)))

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained('../train/T5/t5_base_mg_new/checkpoint-13600')

out = []

exact_match = 0

with open(sys.argv[2], 'w') as fo:
    with torch.no_grad():
        for j in range(len(eng[:2])): 
            
            example = eng[j]
            sol = acts[j]

            input_ids = tokenizer(example, return_tensors="pt").input_ids
            generated_ids = model.generate(input_ids, max_length=512, decoder_start_token_id=tokenizer.bos_token_id)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            print(generated_text, sol)
            exact_match += (generated_text[0].replace(' ', '') == sol.replace(' ', '')) 

            cat = {
                'english': example,
                'generated': generated_text,
                'ground_truth': sol,
            }
                       
            out.append(cat)

        json_object = json.dumps(out, indent = 4)
        fo.write(json_object)
        fo.write('\n')
