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
    train_acc_2 = 0
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
        
        first = output.argmax(dim=1) == training_dataset_labels
        tops = torch.topk(output, 3, dim=1)[1]
        second = torch.tensor([subarr[1].item() for subarr in tops]).to(device)
        third = torch.tensor([subarr[2].item() for subarr in tops]).to(device)
        second_bool = torch.logical_or((second == training_dataset_labels), first)
        last_bool = torch.logical_or((third == training_dataset_labels), second_bool)

        train_acc += first.sum().item()
        train_acc_2 += second_bool.sum().item()
        train_acc_3 += last_bool.sum().item()
        num_examples += batch_size

        if step % loss_computation == 0:
                wandb.log({"Train accuracy Top 1": train_acc / num_examples,
                           "Train accuracy Top 2": train_acc_2 / num_examples
                            "Train accuracy Top 3": train_acc_3 / num_examples})
 
        loss.backward()
        optimizer.step()
    
        model.eval()
        count = 0

        total_loss_val = 0
        total_acc_val = 0
        total_acc_val_2 = 0
        total_acc_val_3 = 0

        if step % 200 == 0:
            for val_input, val_label in val_dataloader:

                output = model(input_ids=batch["input_ids"],
                     labels=batch["labels_input_ids"])
                loss = output.loss
                total_loss_val += loss.item()

                first = (output.argmax(dim=1) == val_label)
                tops = torch.topk(output, 3, dim=1)[1]
                second = torch.tensor([subarr[1].item() for subarr in tops]).to(device)
                third = torch.tensor([subarr[2].item() for subarr in tops]).to(device)
                second_bool = torch.logical_or((second == val_label), first)
                last_bool = torch.logical_or((third == val_label), second_bool)
                

                total_acc_val += first.sum().item()
                total_acc_val_2 += second_bool.sum().item()
                total_acc_val_3 += last_bool.sum().item()
            wandb.log({"Validation Loss":total_loss_val / len(validation_dataset),
                       "Validation Accuracy Top 1": total_acc_val / len(validation_dataset),
                       "Validation Accuracy Top 2": total_acc_val_2 / len(validation_dataset)
                       "Validation Accuracy Top 3": total_acc_val_3 / len(validation_dataset) 
                })
            model.train()
        step += 1

    for val_input, val_label in val_dataloader:

        output = model(input_ids=batch["input_ids"],
                labels=batch["labels_input_ids"])
        loss = output.loss
        total_loss_val += loss.item()

        first = (output.argmax(dim=1) == val_label)
        tops = torch.topk(output, 3, dim=1)[1]
        second = torch.tensor([subarr[1].item() for subarr in tops]).to(device)
        third = torch.tensor([subarr[2].item() for subarr in tops]).to(device)
        second_bool = torch.logical_or((second == val_label), first)
        last_bool = torch.logical_or((third == val_label), second_bool)
        

        total_acc_val += first.sum().item()
        total_acc_val_2 += second_bool.sum().item()
        total_acc_val_3 += last_bool.sum().item()
    wandb.log({"Training Loss in Epoch": train_loss / len(training_dataset),
        "Training Accuracy Top 1 in Epoch": train_acc / len(training_dataset),
        "Training Accuracy Top 2 in Epoch": train_acc_2 / len(training_dataset)
        "Training Accuracy Top 3 in Epoch": train_acc_3 / len(training_dataset) 
        "Validation Loss in Epoch": total_loss_val / len(validation_dataset)
        "Top 1 Validation Accuracy in Epoch": total_acc_val / len(validation_dataset)
    })

    """ print(f'Epoch: {epoch} | Train Loss: {train_loss / len(train_data): .5f} \
                | Train Accuracy: {train_acc / len(train_data): .5f} \
                | Val Loss: {total_loss_val / len(val_data): .5f}\
                | Val Accuracy: {total_acc_val / len(val_data): .5f}') """

    torch.save(model.state_dict(), "t5" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pt")