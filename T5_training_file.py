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
    "layers_freeze": layers_freeze,
}

wandb.init(
    project="mbti_bert_mlm",
    config=dict,
    entity="mbtipredictor"
)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_t5)
model = AutoModelWithLMHead.from_pretrained(model_t5).to(device)

frozen_end_modules = [model.encoder.block[i].layer[0] for i in layers_freeze]
            
for module in frozen_end_modules:
    for param in module.parameters():
        param.requires_grad = False


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


for epoch in range(epochs):
    train_acc = 0
    train_acc_3 = 0
    train_loss = 0
    num_examples = 0
    step = 0
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        output = model(input_ids=batch["input_ids"],
                     labels=batch["labels_input_ids"])
        loss = output.loss
        train_loss += loss.item()

        wandb.log({"Train Loss":loss.item(),
                   "Epoch": epoch,
                   "Step": step})
        loss.backward()
        optimizer.step()
    
        model.eval()
        count = 0

        total_loss_val = 0

        if step % 200 == 0:
            for val_input, val_label in val_dataloader:

                output = model(input_ids=batch["input_ids"],
                     labels=batch["labels_input_ids"])
                loss = output.loss
                total_loss_val += loss.item()
            
            wandb.log({"Validation Loss":total_loss_val / len(validation_dataset),
                })
            model.train()
        step += 1



        """  wandb.log({
        "Traing Loss Epoch": train_loss / len(train_data),
        "Train Accuracy Top 1 Epoch": train_acc / len(train_data),
        "Train Accuracy Top 3 Epoch": train_acc_3 / len(train_data),
        "Val Loss Epoch": total_loss_val / len(val_data),
        "Val Accuracy Epoch": total_acc_val / len(val_data)}
        ) """

    torch.save(model.state_dict(), "t5" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pt")

