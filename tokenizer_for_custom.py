from global_vars import *
import pandas as pd
import numpy as np
import torch
from datasets import Dataset

#pull in data into dataframe
df = pd.read_csv("converted_new_for_custom.csv")
np.random.seed(42)

#tokenize posts
def input_tokenizer(input):
    print("tokenizing input...")
    #pad length of posts to same length
    max_length = input['posts'].str.len().max()
    input['posts'] = input['posts'].str.pad(max_length)
    
    #join all text
    text = []
    for index, row in input.iterrows():
        #print(row)
        #print((row['posts']))
        text.append(row['posts'])
    #create dictionary of char to integer mapping
    chars = set(''.join(text))
    #print(chars)
    int2char = dict(enumerate(chars))
    char2int = {char: ind for ind, char in int2char.items()}
    #print(char2int)

    #tokenize each post
    tokenized = []
    for i in text:
        tokenized.append([char2int[character] for character in i]) #tokenize and add to array
    #print(tokenized)
    #dict_size = len(char2int)
    #seq_len = max_length - 1
    #batch_size = len(text)
    #input_seq = one_hot_encode(tokenized, dict_size, seq_len, batch_size)
    #print("Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(input_seq.shape))
    #print(input_seq)
    return torch.tensor(tokenized)

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

#map personality types to integers
labels = {"INTJ": 0,
          "INTP": 1,
          "ENTJ": 2,
            "ENTP": 3,
            "INFJ": 4,
            "INFP": 5,
            "ENFJ": 6,
            "ENFP": 7,
            "ISTJ": 8,
            "ISFJ": 9,
            "ESTJ": 10,
            "ESFJ": 11,
            "ISTP": 12,
            "ISFP": 13,
            "ESTP": 14,
            "ESFP": 15
          }


def labels_tokenizer(input):
    print("tokenizing labels...")
    tokenized = []
    for i in range(input.shape[0]):
        item = input.iloc[i]
        tokenized.append(labels[item['type']])
    #print(tokenized)
    return torch.tensor(tokenized)

#tokenize input dataframe
def process_input(input):
    return input_tokenizer(input)
def process_labels(labels):
    return labels_tokenizer(labels)

#train val test split
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

#apply to full dataset and save to disk

training_dataset_input = process_input(df_train_input)
validation_dataset_input = process_input(df_val_input)
testing_dataset_input = process_input(df_test_input)

training_dataset_labels = process_labels(df_train_labels)
validation_dataset_labels = process_labels(df_val_labels)
testing_dataset_labels = process_labels(df_test_labels)
#print(testing_dataset_labels)
