import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import sys
from typing import List
from helpers import read_data
import wandb

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
 
import fire
import torch
from datasets import load_dataset
import pandas as pd

 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

eng_dev = []
eng_train = []

acts_dev = []
acts_train = []

with open('../single_goal/6x6worlds/train_set.json') as f:
        data = json.load(f)

        for instance in data:
            eng_train.append(instance['nl_description'])
            acts_train.append(instance['agent_as_a_point'])
    
with open('../single_goal/6x6worlds/dev_set.json') as f:
        data = json.load(f)
        
        for instance in data:
            eng_train.append(instance['nl_description'])
            acts_train.append(instance['agent_as_a_point'])

dataset = [
    {
        "input": eng_train[i],
        "output": acts_train[i]
    }
    for i in range(len(eng_train))
]

BASE_MODEL = "decapoda-research/Llama-7b-hf"
 
model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
 
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
 
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"

def generate_prompt(data_point):
    return f"""Provide a sequence of actions to reach the desired goals.
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < 512
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
 
    result["labels"] = result["input_ids"].copy()
 
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

import json
with open("worlds-dataset.json", "w") as f:
   json.dump(dataset, f)

data = load_dataset("json", data_files="worlds-dataset.json")

train_val = data["train"].train_test_split(
    test_size=0.2, shuffle=False
)
train_data = (
    train_val["train"].map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].map(generate_and_tokenize_prompt)
)

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
 
BATCH_SIZE = 256
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 1500
OUTPUT_DIR = "Llama_single_goal_point"
CUTOFF_LEN = 512

model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

wandb.init(
    # set the wandb project where this run will be logged
    project="llm-planner",
    
    # track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "Llama",
        "dataset": "worlds",
        "epochs": 50,
    }
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, 
    pad_to_multiple_of=8, 
    return_tensors="pt", 
    padding=True
)

training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    num_train_epochs=50,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="wandb"
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)

model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))
 
model = torch.compile(model)
 
trainer.train()
model.save_pretrained(OUTPUT_DIR)