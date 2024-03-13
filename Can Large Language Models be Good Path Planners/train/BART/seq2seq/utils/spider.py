import json
import numpy as np
from typing import Optional
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from .dataset import DataTrainingArguments, normalize
from .trainer import Seq2SeqTrainer, EvalPrediction
import pdb, argparse
import copy
import re

def spider_get_input(question: str, prefix: str)->str:
	# return prefix + question.strip() + " " + refine(serialized_schema).strip()
	return question

def spider_get_target(query: str) -> str:
	# try:
	# 	assert normalize(query).lower() == rejoin_refine_single(refine(normalize(query), replace=True), replace=True)
	# except:
	# 	print(normalize(query).lower(), rejoin_refine_single(refine(normalize(query), replace=True), replace=True))
	# query = copy.deepcopy(refine(normalize(query), replace=True).strip())
	# pdb.set_trace()
	return query

# ------------------------------------------------
# ------------------------------------------------
def difference(str1, str2):
	result1 = ''
	result2 = ''
	maxlen=len(str2) if len(str1)<len(str2) else len(str1)
	#loop through the characters
	for i in range(maxlen):
		#use a slice rather than index in case one string longer than other
		letter1=str1[i:i+1]
		letter2=str2[i:i+1]
		#create string with differences
		if letter1 != letter2:
			result1+=letter1
			result2+=letter2
	return result1

def camel_case_preprocess(s):
    _underscorer1 = re.compile(r'(.)([A-Z][a-z]+)')
    _underscorer2 = re.compile('([a-z0-9])([A-Z])')
    subbed = _underscorer1.sub(r'\1\2', s)
    return _underscorer2.sub(r'\1# \2', subbed).lower()

def camel_case_postprocess(data):
    data_split = data.split()
    processed = ""
    remove = False
    for line in data_split:
        if '#' in line:
            processed += " "+ line.replace('#','')
            remove = True
        else:
            if remove:
                processed += line
                remove = False
            else:
                processed += " "+ line
    return processed

def refine(raw_data, replace = False):
    data = copy.deepcopy(raw_data)
    if replace == True:
        data = data.replace("asc (", "ascend (")
        data = data.replace("desc (", "descending (")
        data = data.replace("asc(", "ascend (")
        data = data.replace("desc(", "descending (")
        data = data.replace("avg(", "average (")
        data = data.replace("avg (", "average (")
    data_split = data.split()
    refined_word = " "
    for word in data_split:
        # if "struct_sep" in word:
        #    continue
        if "_" not in word and "." not in word:
            refined_word += " " + word
        else:   
            if "_" in word:
                w = word.split("_")
                rw1 = " " + " _ ".join(w)
                if len(rw1.split(".")) == 1:
                    refined_word += rw1
                else:
                    temp_refined_word = rw1.split(".")
                    refined_word +=" "+" . ".join(temp_refined_word)
            if "." in word and "_" not in word:
                w = word.split(".")
                refined_word +=" "+" . ".join(w)
    return refined_word.strip()

def rejoin_refine_single(raw_data, replace=True):
	keywords = ('except_', 'intersect_', 'union_')
	# camel_data = camel_case_postprocess(raw_data)
	camel_data = raw_data
	if replace == True:
		camel_data = camel_data.replace("ascend (", "asc (")
		camel_data = camel_data.replace("ascend(", "asc (")
		camel_data = camel_data.replace("descending (", "desc (")
		camel_data = camel_data.replace("descending(", "desc (")
		camel_data = camel_data.replace("average(", "avg (")
		camel_data = camel_data.replace("average (", "avg (")
	data_split = camel_data.split()
	refined_data = ""
	remove = False
	for i, word in enumerate(data_split):
		if "_" not in word and "." not in word:
			if remove:
				refined_data += word
				remove = False
			else:
				refined_data += " " + word
		else:
			refined_data += word
			if data_split[i-1] + word not in keywords:
				remove = True
	return refined_data.strip()

def rejoin_refine(data):
    refined_data = []
    for d in data:
        refined_data.append(rejoin_refine_single(refine(d)))
    return refined_data

def refine_metas(data):
	r_data = []
	for d in data:
		d['query'] = d['query']
		d['context'] = d['context']
		d['label'] = rejoin_refine_single(d['label'])
		r_data.append(d)
	return r_data

# ---------------------------------------------

# def spider_add_serialized_schema(ex: dict, data_training_args: DataTrainingArguments)->dict:
# 	serialized_schema = serialize_schema(question = ex["question"], db_path=ex["db_path"], db_id = ex["db_id"], db_column_names = ex["db_column_names"], db_table_names = ex["db_table_names"], schema_serialization_type=data_training_args.schema_serialization_type, schema_serialization_randomized=data_training_args.schema_serialization_randomized, schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id, schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_id, normalize_query=data_training_args.normalize_query)
# 	return {"serialized_schema": serialized_schema}

def spider_pre_process_function(batch: dict, max_source_length: Optional[int], max_target_length: Optional[int], data_training_args: DataTrainingArguments, tokenizer: PreTrainedTokenizerBase)->dict:
	# pdb.set_trace()
	prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else ""
	inputs = [
	spider_get_input(question, prefix) for question in batch["question"]
	]
	# pdb.set_trace()
	model_inputs: dict = tokenizer(inputs, max_length=max_source_length, padding = False, truncation = True, return_overflowing_tokens = False)
	targets = [spider_get_target(query) for query in batch["target"]]
	with tokenizer.as_target_tokenizer():
		labels = tokenizer(targets, max_length=max_target_length, padding=False, truncation=True, return_overflowing_tokens = False)
	model_inputs["labels"] = labels["input_ids"]
	return model_inputs

class SpiderTrainer(Seq2SeqTrainer):
	def _post_process_function(self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str)->EvalPrediction:
		# pdb.set_trace()
		inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens = True)
		label_ids = [f["labels"] for f in features]
		if self.ignore_pad_token_for_loss:
			_label_ids = np.where(label_ids!=-100, label_ids, self.tokenizer.pad_token_id)
		decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
		metas = [
		{
			"target": x["target"],
			"question": x["question"],
			"context": context,
			"label": label,
		}
		for x, context, label in zip(examples, inputs, decoded_label_ids)
		]
		predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
		assert len(metas) == len(predictions)
		with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
			json.dump(
				[dict(**{"prediction": prediction}, **meta) for prediction, meta in zip(predictions, metas)],
				f,
				indent=4,
			)
		return EvalPrediction(predictions=predictions, label_ids=label_ids, metas=metas)

	def _compute_metrics(self, eval_prediction: EvalPrediction)->dict:
		predictions, label_ids, metas = eval_prediction
		# pdb.set_trace()
		# predictions = self.remove_special_token(predictions)
		references = metas
		return self.metric.compute(predictions = predictions, references = references)
	
	# def construct_hyper_param(self):
	# 	parser = argparse.ArgumentParser()
	# 	args = parser.parse_args()
	# 	return args




