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
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, index):
        input_text = '<s> ' + self.input_texts[index]
        target_text = self.target_texts[index]

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

def compute_exact_match(predictions, references):
    em = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    em_score = em / len(predictions)
    return em_score

def compute_metrics(eval_prediction):
    predictions = tokenizer.batch_decode(eval_prediction.predictions, skip_special_tokens=True)
    references = tokenizer.batch_decode(eval_prediction.label_ids, skip_special_tokens=True)
    return {"exact_match": compute_exact_match(predictions, references)}

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
        output_dir='./T5_base_single_goal_point',          
        num_train_epochs=200,              # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        warmup_steps=0,                # number of warmup steps for learning rate scheduler
        warmup_ratio=0.0,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,
        save_total_limit = 250,
        include_inputs_for_metrics = True,
        save_steps=500,
        learning_rate=1e-4,
        evaluation_strategy="steps",
        eval_steps=500,
        report_to="wandb",
        adam_epsilon=1e-6,
        adafactor=False,
        label_smoothing_factor=0.0,
        lr_scheduler_type="constant"
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