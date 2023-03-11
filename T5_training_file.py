from global_vars import *
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm import tqdm
import pandas as pd
import wandb
import pickle

dict = {
    "model_name": model_t5,
    "learning_rate": learning_rate,
    "max_length_input": max_length_input,
    "doTruncate": doTruncate,
    "doPadding": doPadding,
    "split_train_test": split_train_test,
    "split_train_val": split_train_val,
    "batch_size": batch_size,
    "epochs": epochs,
    "freeze_threshold": freeze_threshold,
}

wandb.init(
    project="mbti_bert_mlm",
    config=dict,
    entity="mbtipredictor"
)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_t5)
model = AutoModelWithLMHead.from_pretrained(model_t5).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

training_dataset_input = load_from_disk("dataset/train_dataset_input")
training_dataset_labels = load_from_disk("dataset/train_dataset_labels")
validation_dataset_input = load_from_disk("dataset/validation_dataset_input")
validation_dataset_labels = load_from_disk("dataset/validation_dataset_labels")

training_dataset = Dataset.from_dict({"input_ids": training_dataset_input['input_ids'], #train input ids
                             "labels_input_ids": training_dataset_labels['input_ids'] #label input ids
                             }).with_format("torch", device=device)

validation_dataset = Dataset.from_dict({"input_ids": validation_dataset_input['input_ids'], #train input ids
                             "labels_input_ids": validation_dataset_labels['input_ids'] #label input ids
                             }).with_format("torch", device=device)
# do testing dataset in ipynb file

train_dataloader = DataLoader(training_dataset, batch_size=batch_size, drop_last=False)
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, drop_last=False)

step = -1

for epoch in range(epochs):
    train_acc = 0
    train_loss = 0
    step = 0
    for batch in tqdm(train_dataloader):
        step += 1
        optimizer.zero_grad()
        loss = model(input_ids=batch["input_ids"],
                     labels=batch["labels_input_ids"]).loss
        train_loss += loss.item()

        wandb.log({"train loss":loss.item(),
                   "batch_index": step,
                   "epoch": epoch,
                   "step": step})
        loss.backward()
        optimizer.step()

    
    model.eval()
    total_acc_val = 0
    total_loss_val = 0
    count = 0

    for val_batch in val_dataloader:
        loss = model(input_ids=val_batch["input_ids"],
                        labels=val_batch["labels_input_ids"]).loss # compute loss
        total_loss_val += loss.item()
        count += 1

    print(f'Epoch: {epoch} | Train Loss: {train_loss / len(training_dataset): .5f} \
                | Train Accuracy: {train_acc / len(training_dataset): .5f} \
                | Val Loss: {total_loss_val / len(validation_dataset): .5f}\
                | Val Accuracy: {total_acc_val / len(validation_dataset): .5f}')
    wandb.log({"validation loss":total_loss_val / len(validation_dataset),
                "validation accuracy":total_acc_val / len(validation_dataset),
                    "train accuracy":train_acc / len(training_dataset),
                "step": step})
    model.train()
