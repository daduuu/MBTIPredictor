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


dict = {
    "model_name": model_name,
    "learning_rate": learning_rate,
    "max_length_input": max_length_input,
    "doTruncate": doTruncate,
    "doPadding": doPadding,
    "split_train_test": split_train_test,
    "split_train_val": split_train_val,
    "batch_size": batch_size,
    "epochs": epochs,
    "freeze_threshold": freeze_threshold,
    "layers_freeze": layers_freeze
}

wandb.init(
    project="mbti_bert_mlm",
    config=dict,
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
df = df[:1000]

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

            for i in layers_freeze:
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
            train_acc_3 = 0
            train_loss = 0
            num_examples = 0
            step = 0
            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                loss_batch = loss_func(output, train_label.long())
                train_loss += loss_batch.item()
                wandb.log({"Train Loss":loss_batch.item(),
                   "Epoch": epoch,
                   "Step": step})
                
                
    
                first = output.argmax(dim=1) == train_label
                tops = torch.topk(output, 3, dim=1)[1]
                second = torch.tensor([subarr[1].item() for subarr in tops]).to(device)
                third = torch.tensor([subarr[2].item() for subarr in tops]).to(device)
                second_bool = torch.logical_or((second == train_label), first)
                last_bool = torch.logical_or((third == train_label), second_bool)

                train_acc += first.sum().item()
                train_acc_3 += last_bool.sum().item()
                num_examples += batch_size

                if step % loss_computation == 0:
                        wandb.log({"Train accuracy Top 1": train_acc / num_examples,
                                    "Train accuracy Top 3": train_acc_3 / num_examples})

                model.zero_grad()
                loss_batch.backward()
                optimizer.step()
            
                total_acc_val = 0
                total_loss_val = 0
                total_acc_val_3 = 0


                if step % 200 == 0:
                    for val_input, val_label in val_dataloader:

                        val_label = val_label.to(device)
                        mask = val_input['attention_mask'].to(device)
                        input_id = val_input['input_ids'].squeeze(1).to(device)

                        output = model(input_id, mask)

                        loss_batch = loss_func(output, val_label.long())
                        total_loss_val += loss_batch.item()
                        
                        first = output.argmax(dim=1) == val_label
                        tops = torch.topk(output, 3, dim=1)[1]
                        second = torch.tensor([subarr[1].item() for subarr in tops]).to(device)
                        third = torch.tensor([subarr[2].item() for subarr in tops]).to(device)
                        second_bool = torch.logical_or((second == val_label), first)
                        last_bool = torch.logical_or((third == val_label), second_bool)
                

                        total_acc_val += first.sum().item()
                        total_acc_val_3 += last_bool.sum().item()
                    wandb.log({"Validation Loss":total_loss_val / len(val_data),
                            "Validation Accuracy Top 1":total_acc_val / len(val_data),
                                "Validation Accuracy Top 3":total_acc_val_3 / len(val_data),
                        })
                    model.train()
                step += 1

            total_acc_val = 0
            total_loss_val = 0
            total_acc_val_3 = 0
            for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    loss_batch = loss_func(output, val_label.long())
                    total_loss_val += loss_batch.item()
                    
                    first = output.argmax(dim=1) == val_label
                    tops = torch.topk(output, 3, dim=1)[1]
                    second = torch.tensor([subarr[1].item() for subarr in tops]).to(device)
                    third = torch.tensor([subarr[2].item() for subarr in tops]).to(device)
                    second_bool = torch.logical_or((second == val_label), first)
                    last_bool = torch.logical_or((third == val_label), second_bool)

                    total_acc_val += first.sum().item()
                    total_acc_val_3 += last_bool.sum().item()

            wandb.log({
                "Traing Loss Epoch": train_loss / len(train_data),
                "Train Accuracy Top 1 Epoch": train_acc / len(train_data),
                "Train Accuracy Top 3 Epoch": train_acc_3 / len(train_data),
                "Val Loss Epoch": total_loss_val / len(val_data),
                "Val Accuracy Epoch": total_acc_val / len(val_data)}
            )
        

                
            
            """ print(f'Epoch: {epoch} | Train Loss: {train_loss / len(train_data): .5f} \
                | Train Accuracy: {train_acc / len(train_data): .5f} \
                | Val Loss: {total_loss_val / len(val_data): .5f}\
                | Val Accuracy: {total_acc_val / len(val_data): .5f}') """
            
                
                
            
    torch.save(model.state_dict(), "bert_mlm2" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pt")
                  

model = BertForClassification(freeze=True)

              
train(model, df_train, df_val, learning_rate, epochs)