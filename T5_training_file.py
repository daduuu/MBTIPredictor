from global_vars import *
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm import tqdm
import pandas as pd
import wandb
import pickle
from torch import nn

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

base_model = AutoModelWithLMHead.from_pretrained(model_t5).to(device)
base_model.train()

optimizer = torch.optim.AdamW(base_model.parameters(), lr=learning_rate)

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

class T5ForClassification(nn.Module):

    def __init__(self, dropout=0.5, freeze=False):

        super(T5ForClassification, self).__init__()

        self.model = AutoModelWithLMHead.from_pretrained(base_model)
        self.freeze = freeze

        #Freeze Last freeze_threshold Layers
        if self.freeze:
            #set freeze_threshold to be positive (postive # of layers from end to freese)
            modules_to_freeze = modules_to_freeze = [model.encoder.block[i].layer[0] for i in range(len(model.encoder.block) - 1, len(model.encoder.block) - freeze_threshold -1, -1)]
            
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

model = T5ForClassification(freeze=True)

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

