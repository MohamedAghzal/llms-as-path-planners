from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from transformers import BertTokenizer, EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder, Seq2SeqTrainer, TrainingArguments
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from evaluate import load
import transformers.data
import torch
import json
import os


eng = []
acts = []
locs = [] 
n_obs = []
paths = []

with open('../single_goal/6x6worlds/test_seen.json') as f:
    data = json.load(f)

    for data_point in data:
        eng.append(data_point['nl_description'])
        acts.append(data_point['agent_has_direction'])



print(len(set(eng)))

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained('../train/BART_base_single_goal_direction/checkpoint-40000')
out = []

exact_match = 0

with open('out_bart_base_sg_direction_seen.json', 'w') as fo:
    with torch.no_grad():
        for j in range(len(eng)): 
            
            example = eng[j]
            sol = acts[j]

            input_ids = tokenizer(example, return_tensors="pt").input_ids
            generated_ids = model.generate(input_ids, max_length=512, decoder_start_token_id=tokenizer.bos_token_id)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            print(generated_text)
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
        print(exact_match / len(input_english))
