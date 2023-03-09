from global_vars import *
import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer

import pandas as pd
import pickle
import global_vars
import numpy as np


tokenizer = AutoTokenizer.from_pretrained(model_name)
df = pd.read_csv("converted_new.csv")

np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(split_train_test*len(df)), int(split_train_val*len(df))])

labels = {0:0, 1:1}

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['type']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 100, truncation=True,
                                return_tensors="pt") for text in df['posts']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

ds_train = Dataset(df_train)
ds_test = Dataset(df_test)
ds_val = Dataset(df_val)

with open("ds_train.pkl", "wb") as f:
    pickle.dump(ds_train, f)
with open("ds_test.pkl", "wb") as f:
    pickle.dump(ds_test, f)
with open("ds_val.pkl", "wb") as f:
    pickle.dump(ds_val, f)