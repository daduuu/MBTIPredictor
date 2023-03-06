from global_vars import *
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
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

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

file = open("input_encoding_bert_final.pkl", 'rb')
input_encoding = pickle.load(file)
file = open("output_encoding_bert_final.pkl", 'rb')
output_encoding = pickle.load(file)



input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
labels = output_encoding["input_ids"]

labels[labels == tokenizer.pad_token_id] = -100
labels = torch.where((input_ids == tokenizer.mask_token_id) & (input_ids  != 101) & (input_ids  != 102), labels, -100)

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