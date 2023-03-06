from global_vars import *
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import Dataset

import pandas as pd
import pickle


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
df = pd.read_csv("converted.csv")
df['input'] = "Given: " + df['posts'] + "MBTI: " + tokenizer.mask_token
df['output'] = "Given: " + df['posts'] + " MBTI: " + df['type']

df = df[200000:]
input_encoding = tokenizer(
    [str(s) for s in df["input"].tolist()],
    padding=doPadding,
    max_length=max_length_input,
    truncation=doTruncate,
    return_tensors="pt",
)

output_encoding = tokenizer(
    [str(s) for s in df["output"].tolist()],
    padding=doPadding,
    max_length=max_length_input,
    truncation=doTruncate,
    return_tensors="pt",
)

with open('input_encoding_bert2.pkl', 'wb') as fp:
    pickle.dump(input_encoding, fp)
    print('dictionary saved successfully to file')

with open('output_encoding_bert2.pkl', 'wb') as fp:
    pickle.dump(input_encoding, fp)
    print('dictionary saved successfully to file')

