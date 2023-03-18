from global_vars import *
import torch
import pandas as pd
import pickle
from transformers import AutoTokenizer
import numpy as np
from datasets import Dataset
from datetime import datetime

tokenizer = AutoTokenizer.from_pretrained(model_t5)

df = pd.read_csv("converted_new_for_T5.csv")
time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


np.random.seed(42)


df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(split_train_test*len(df)), int(split_train_val*len(df))])

df_train_input = df_train['posts'].to_frame()
df_train_labels = df_train['type'].to_frame()

df_val_input = df_val['posts'].to_frame()
df_val_labels = df_val['type'].to_frame()

df_test_input = df_test['posts'].to_frame()
df_test_labels = df_test['type'].to_frame()

ds_train_input = Dataset.from_pandas(df_train_input)
ds_train_labels = Dataset.from_pandas(df_train_labels)

ds_val_input = Dataset.from_pandas(df_val_input)
ds_val_labels = Dataset.from_pandas(df_val_labels)

ds_test_input = Dataset.from_pandas(df_test_input)
ds_test_labels = Dataset.from_pandas(df_test_labels)



def process_input(input):
    return tokenizer(input['posts'], padding=doPadding, max_length=max_length_input, truncation=doTruncate, return_tensors="pt")
def process_labels(labels):
    return tokenizer(labels['type'], padding=doPadding, max_length=4, truncation=doTruncate, return_tensors="pt")

training_dataset_input = ds_train_input.map(process_input, batched = True)
validation_dataset_input = ds_val_input.map(process_input, batched = True)
testing_dataset_input = ds_test_input.map(process_input, batched= True)

training_dataset_labels = ds_train_labels.map(process_labels, batched=True)
validation_dataset_labels = ds_val_labels.map(process_labels, batched=True)
testing_dataset_labels = ds_test_labels.map(process_labels, batched=True)

training_dataset_input.save_to_disk("dataset" + time + "/train_dataset_input")
validation_dataset_input.save_to_disk("dataset" + time + "/validation_dataset_input")
testing_dataset_input.save_to_disk("dataset" + time + "/testing_dataset_input")

training_dataset_labels.save_to_disk("dataset" + time + "/train_dataset_labels")
validation_dataset_labels.save_to_disk("dataset" + time + "/validation_dataset_labels")
testing_dataset_labels.save_to_disk("dataset" + time + "/testing_dataset_labels")