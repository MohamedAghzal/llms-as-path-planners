import torch
from torch.utils.data import Dataset
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import T5Config
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from evaluate import load
import transformers.data
from torch.utils.data import Dataset
import json
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb

class Seq2SeqDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer, max_input_length=512, max_target_length=512):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.input_texts = input_texts
        self.target_texts = target_texts

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        source_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        # Tokenize the input and target text
        inputs = self.tokenizer.encode_plus(
            source_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        targets = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Extract input_ids, attention_mask, and target_ids
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        target_ids = targets['input_ids'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_ids': target_ids,
        }

def main():
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="llm-planner",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 5e-5,
        "architecture": "T5-base",
        "dataset": "worlds",
        "epochs": 100,
        }
    )

    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    config = T5Config(
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_max_length=512
    )

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model.bos_token_id=tokenizer.bos_token_id
    model.eos_token_id=tokenizer.eos_token_id
    model.decoder_max_length=512
    model.vocab_size=len(tokenizer)

    nl = []
    actions = []

    with open('../single_goal/6x6worlds/train_set.json') as f:
        data = json.load(f)

        for instance in data:
            nl.append(instance['nl_description'])
            actions.append(instance['agent_as_a_point'])
    
    nl_dev = []
    actions_dev = []

    with open('../single_goal/6x6worlds/dev_set.json') as f:
        data = json.load(f)
        
        for instance in data:
            nl_dev.append(instance['nl_description'])
            actions_dev.append(instance['agent_as_a_point'])

    print(len(set(nl)))

    train_dataset = Seq2SeqDataset(nl, actions, tokenizer=tokenizer)
    dev_set = Seq2SeqDataset(nl_dev, actions_dev, tokenizer=tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir='./T5_base_single_goal_point_ds',          
        num_train_epochs=80,              # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,
        save_total_limit = 250,
        include_inputs_for_metrics = True,
        save_steps=1000,
        learning_rate=1e-4,
        evaluation_strategy="steps",
        eval_steps=1000,
        report_to="wandb",
        fp16=True
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