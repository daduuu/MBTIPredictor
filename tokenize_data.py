from global_vars import *
import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer
from datasets import Dataset

import pandas as pd
import pickle
import global_vars
import numpy as np


tokenizer = AutoTokenizer.from_pretrained(model_name)
df = pd.read_csv("converted.csv")


ds = Dataset.from_pandas(df[:10000])
ending_names = ["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"]
def preprocess_function(examples):
    first_sentences = [[context] * 16 for context in examples["posts"]]

    question_headers = examples['pad']
    second_sentences = [[f"{header} {end}".strip() for end in ending_names] for i, header in enumerate(question_headers)]
    
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=doTruncate, padding=doPadding, max_length=max_length_input)

    return {k: [v[i:i+16] for i in range(0, len(v), 16)] for k, v in tokenized_examples.items()}


tokenized_inputs = ds.map(preprocess_function, batched=True)

with open('input_encoding_bert_small.pkl', 'wb') as fp:
    pickle.dump(tokenized_inputs, fp)
    print('dictionary saved successfully to file')

