import os
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import BartConfig
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from evaluate import load
import transformers.data
from torch.utils.data import Dataset
import json
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb

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

class Seq2SeqDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer, max_input_length=1024, max_target_length=1024):
        self.tokenizer = tokenizer
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, index):
        input_text = self.input_texts[index]
        target_text = '<s> ' + self.target_texts[index]

        # Tokenize input and target sequences
        input_tokens = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        target_tokens = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Return the tokenized input and target sequences
        return {
            'input_ids': input_tokens['input_ids'].squeeze(),
            'attention_mask': input_tokens['attention_mask'].squeeze(),
            'decoder_input_ids': target_tokens['input_ids'].squeeze()[:-1],
            'decoder_attention_mask': target_tokens['attention_mask'].squeeze()[:-1],
            'labels': target_tokens['input_ids'].squeeze()[1:]
        }

def main():
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="llm-planner",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 5e-5,
        "architecture": "bart-base",
        "dataset": "worlds",
        "epochs": 100,
        }
    )

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    config = BartConfig(
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_max_length=1024
    )

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    model.bos_token_id=tokenizer.bos_token_id
    model.eos_token_id=tokenizer.eos_token_id
    model.decoder_max_length=1024
    model.vocab_size=len(tokenizer)

    nl = []
    actions = []

    with open('../single_goal/6x6worlds/train_set.json') as f:
        data = json.load(f)

        for instance in data:
            nl.append(instance['nl_description'])
            actions.append(solution_direction(instance['agent_as_a_point']))
    
    nl_dev = []
    actions_dev = []

    with open('../single_goal/6x6worlds/dev_set.json') as f:
        data = json.load(f)
        
        for instance in data:
            nl_dev.append(instance['nl_description'])
            actions_dev.append(solution_direction(instance['agent_as_a_point']))

    print(len(set(nl)))

    train_dataset = Seq2SeqDataset(nl, actions, tokenizer=tokenizer)
    dev_set = Seq2SeqDataset(nl_dev, actions_dev, tokenizer=tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir='./BART_base_single_goal_direction',          
        num_train_epochs=80,              # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,
        save_total_limit = 128,
        include_inputs_for_metrics = True,
        save_steps=200,
        learning_rate=5e-5,
        evaluation_strategy="steps",
        eval_steps=200,
        report_to="wandb"
    )

    model.resize_token_embeddings(len(tokenizer))

    trainer = Seq2SeqTrainer(
        model=model,                            
        args=training_args,            
        train_dataset=train_dataset,
        eval_dataset=dev_set
    )

    trainer.train()
if __name__ == "__main__":
    main()
