from global_vars import *
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import Dataset

import pandas as pd
import pickle
import global_vars
import numpy as np


tokenizer = AutoTokenizer.from_pretrained(model_name)
df = pd.read_csv("converted.csv")


ds = Dataset.from_pandas(df)

def preprocess_data(examples):
    try:
        return tokenizer(examples["posts"], truncation=doTruncate, padding=doPadding, max_length=max_length_input)
    except:

        raise Exception("Error in preprocessing data " + str(examples['new_col']))


tokenized_inputs = ds.map(preprocess_data, batched=True)

with open('input_encoding_bert.pkl', 'wb') as fp:
    pickle.dump(tokenized_inputs, fp)
    print('dictionary saved successfully to file')

