from global_vars import *
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, T5ForConditionalGeneration
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm import tqdm
import pandas as pd
import wandb
import pickle
import sys
from datetime import datetime


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model_saved = "t52023-03-18-13-13-00.pt"
tokenizer = AutoTokenizer.from_pretrained(model_t5)
model = T5ForConditionalGeneration.from_pretrained(model_t5).to(device)
model.load_state_dict(torch.load(model_saved, map_location=torch.device('cpu')))

def compute_accuracy(preds, labels):
    o2 = [tokenizer.decode(preds[i], skip_special_tokens=True) for i in range(preds.shape[0])]
    b3 = [tokenizer.decode(labels[i], skip_special_tokens=True) for i in range(labels.shape[0])]
    count = 0
    for i in range(len(o2)):
        if o2[i].lower().strip() == b3[i].lower().strip():
            count += 1
    return count

time = "2023-03-18-04-46-28"
testing_dataset_input = load_from_disk("dataset" + time + "/testing_dataset_input")
testing_dataset_labels = load_from_disk("dataset" + time + "/testing_dataset_labels")

testing_dataset = Dataset.from_dict({"input_ids": testing_dataset_input['input_ids'], #train input ids
                                "attention_mask": testing_dataset_input['attention_mask'],
                             "labels_input_ids": testing_dataset_labels['input_ids'] #label input ids
                             }).with_format("torch")
test_dataloader = DataLoader(testing_dataset, batch_size=batch_size, drop_last=False)

val_acc = 0
for batch in tqdm(test_dataloader):
    output = model(input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                     labels=batch["labels_input_ids"].to(device))
    outs2 = model.generate(batch["input_ids"].to(device), max_new_tokens=4).to(device)
    val_acc += compute_accuracy(outs2, batch["labels_input_ids"])

print(val_acc / len(testing_dataset))