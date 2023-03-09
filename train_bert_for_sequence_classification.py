from global_vars import *
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm import tqdm
import pandas as pd
import wandb
from torch import nn
from torch.optim import Adam
import pickle
import numpy as np
from datetime import datetime
import global_vars

wandb.init(
    project="mbti_bert_mlm",
    entity="mbtipredictor"
)

labels = {0: 0,
          1: 1,
          2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11,
            12: 12,
            13: 13,
            14: 14,
            15: 15
          }
df = pd.read_csv("converted_new.csv")

tokenizer = AutoTokenizer.from_pretrained(model_name)

df = df.sample(frac = 1, random_state = 42)

np.random.seed(42)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(split_train_test*len(df)), int(split_train_val*len(df))])


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['type']]
        self.posts = [tokenizer(post, 
                               padding=doPadding, max_length = max_length_input, truncation=doTruncate,
                                return_tensors="pt") for post in df['posts']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return self.posts[idx], np.array(self.labels[idx])


class BertForClassification(nn.Module):

    def __init__(self, dropout=0.5, freeze=False):

        super(BertForClassification, self).__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 16)
        self.relu = nn.ReLU()
        self.freeze = freeze

        #Freeze Last freeze_threshold Layers
        if self.freeze:
            for param in self.model.embeddings.parameters():
                if(param.requires_grad):
                    param.requires_grad = False


            for i in range(len(self.model.encoder.layer)):
                if(i < len(self.model.encoder.layer) + freeze_threshold):
                    for param in self.model.encoder.layer[i].parameters():
                        if(param.requires_grad):
                            param.requires_grad = False

    def forward(self, input_id, mask):

        _, o1 = self.model(input_ids= input_id, attention_mask=mask, return_dict=False)
        o2 = self.dropout(o1)
        o3 = self.linear(o2)
        fo = self.relu(o3)

        return fo

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    model = model.to(device)
    loss_func = loss_func.to(device)

    
    for epoch in range(epochs):

            train_acc = 0
            train_loss = 0
            step = 0
            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                loss_batch = loss_func(output, train_label.long())
                train_loss += loss_batch.item()

                wandb.log({"train loss":loss_batch.item(),
                   "batch_index": step,
                   "epoch": epoch,
                   "step": step})
                
                train_acc += (output.argmax(dim=1) == train_label).sum().item()

                model.zero_grad()
                loss_batch.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():
                c = 1

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    loss_batch = loss_func(output, val_label.long())
                    total_loss_val += loss_batch.item()
                    
                    total_acc_val = (output.argmax(dim=1) == val_label).sum().item()
                    c += 1

                
            
            print(f'Epoch: {epoch} | Train Loss: {train_loss / len(train_data): .5f} \
                | Train Accuracy: {train_acc / len(train_data): .5f} \
                | Val Loss: {total_loss_val / len(val_data): .5f}\
                | Val Accuracy: {total_acc_val / len(val_data): .5f}')
            wandb.log({"validation loss":total_loss_val / len(val_data),
                       "step": step})
            
            step += 1
    torch.save(model.state_dict(), "bert_mlm2" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pt")
                  

model = BertForClassification(freeze=True)

              
train(model, df_train, df_val, learning_rate, epochs)