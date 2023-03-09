from global_vars import *
import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer
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

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

file = open("input_encoding_bert_small.pkl", 'rb')
input_encoding = pickle.load(file)
input_encoding = input_encoding[:1000]

labels = pd.read_csv("converted.csv")['type']
labels = labels[:1000]



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

torch.save(model.state_dict(), "mbti_bert_small.pt")