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
    "layers_freeze": layers_freeze,
}

wandb.init(
    project="mbti_bert_mlm",
    config=dict,
    entity="mbtipredictor"
) 

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_t5)
model = T5ForConditionalGeneration.from_pretrained(model_t5).to(device)

frozen_end_modules = [model.encoder.block[i].layer[0] for i in layers_freeze]
            
for module in frozen_end_modules:
    for param in module.parameters():
        param.requires_grad = False


def compute_accuracy(preds, labels):
    o2 = [tokenizer.decode(preds[i], skip_special_tokens=True) for i in range(preds.shape[0])]
    b3 = [tokenizer.decode(labels[i], skip_special_tokens=True) for i in range(labels.shape[0])]
    count = 0
    for i in range(len(o2)):
        if o2[i].lower().strip() == b3[i].lower().strip():
            count += 1
    return count


model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

time = "2023-03-18-04-46-28"

training_dataset_input = load_from_disk("dataset" + time + "/train_dataset_input")
training_dataset_labels = load_from_disk("dataset" + time + "/train_dataset_labels")
validation_dataset_input = load_from_disk("dataset" + time + "/validation_dataset_input")
validation_dataset_labels = load_from_disk("dataset" + time + "/validation_dataset_labels")

training_dataset = Dataset.from_dict({"input_ids": training_dataset_input['input_ids'], #train input ids
                                "attention_mask": training_dataset_input['attention_mask'],
                             "labels_input_ids": training_dataset_labels['input_ids'] #label input ids
                             }).with_format("torch", device=device)

validation_dataset = Dataset.from_dict({"input_ids": validation_dataset_input['input_ids'], #train input ids
                        "attention_mask": validation_dataset_input['attention_mask'],
                             "labels_input_ids": validation_dataset_labels['input_ids'] #label input ids
                             }).with_format("torch", device=device)
# do testing dataset in ipynb file

train_dataloader = DataLoader(training_dataset, batch_size=batch_size, drop_last=False)
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, drop_last=False)


for epoch in range(epochs):
    train_acc = 0
    train_loss = 0
    num_examples = 0
    step = 0
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        output = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                     labels=batch["labels_input_ids"])
        loss = output.loss
        train_loss += loss.item()

        
        outs = model.generate(batch["input_ids"], max_new_tokens=4)
        train_acc += compute_accuracy(outs, batch["labels_input_ids"])
        num_examples += batch_size

        wandb.log({"Train Loss":loss.item(),
                   "Epoch": epoch,
                   "Step": step,
                   "Train Acc: ": train_acc / num_examples})
 
        loss.backward()
        optimizer.step()
    
        model.eval()
        count = 0
        val_loss = 0
        val_acc = 0

        if step % 10607 == 0:
            for val_batch in val_dataloader:
                output = model(input_ids=val_batch["input_ids"],
                    attention_mask=val_batch["attention_mask"],
                     labels=val_batch["labels_input_ids"])
                loss = output.loss
                val_loss += loss.item()

                outs2 = model.generate(val_batch["input_ids"], max_new_tokens=4)
                val_acc += compute_accuracy(outs2, val_batch["labels_input_ids"])

            wandb.log({"Val Loss": val_loss / len(validation_dataset),
                "Val Acc: ": val_acc / len(validation_dataset)})

        total_loss_val = 0


        step += 1


torch.save(model.state_dict(), "t5" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pt")
print(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
