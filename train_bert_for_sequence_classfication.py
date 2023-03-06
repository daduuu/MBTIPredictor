from global_vars import *
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm import tqdm
import pandas as pd
import wandb
import pickle

wandb.init(
    project="mbti_bert_mlm",
    entity="mbtipredictor"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

m1 = {0: "ISTJ", 1: "ISFJ", 2: "INFJ", 3: "INTJ", 4: "ISTP", 5: "ISFP", 6: "INFP", 7: "INTP", 8: "ESTP", 9: "ESFP", 10: "ENFP", 11: "ENTP", 12: "ESTJ", 13: "ESFJ", 14: "ENFJ", 15: "ENTJ"}
m2 = {"ISTJ" : 0, "ISFJ" : 1, "INFJ" : 2, "INTJ" : 3, "ISTP" : 4, "ISFP" : 5, "INFP" : 6, "INTP" : 7, "ESTP" : 8, "ESFP" : 9, "ENFP" : 10, "ENTP" : 11, "ESTJ" : 12, "ESFJ" : 13, "ENFJ" : 14, "ENTJ" : 15}


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=16, id2label=m1, label2id=m2).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

file = open("input_encoding_bert.pkl", 'rb')
input_encoding = pickle.load(file)

labels = pd.read_csv("converted.csv")['type']



input_ids, attention_mask = input_encoding['input_ids'], input_encoding['attention_mask']

dataset = Dataset.from_dict({"input_ids": input_ids,
                             "attention_mask": attention_mask,
                             "labels": labels
                             }).with_format("torch", device=device)

temp_dataset = dataset.train_test_split(test_size=1 - split_train_test, shuffle=True, seed=42)
test_dataset = temp_dataset["test"]

temp2_dataset = temp_dataset["train"].train_test_split(test_size = 1 - split_train_val, shuffle=True, seed=42)

train_dataset = temp2_dataset["train"]
val_dataset = temp2_dataset["test"]


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False)


step = -1

for epoch in range(epochs):
    for i, batch in tqdm(enumerate(train_dataloader)):
        step += 1
        optimizer.zero_grad()
        loss = model(input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"],
                     labels=batch["labels"]).loss

        wandb.log({"train loss":loss.item(),
                   "batch_index": i,
                   "epoch": epoch,
                   "step": step})
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            model.eval()
            val_loss = 0
            count = 0
            for j, val_batch in enumerate(val_dataloader):
                loss = model(input_ids=val_batch["input_ids"],
                             attention_mask=val_batch["attention_mask"],
                             labels=val_batch["labels"]).loss # compute loss
                val_loss += loss.item()
                count += 1

            wandb.log({"validation loss": val_loss / count,
                       "step": step})
            model.train()