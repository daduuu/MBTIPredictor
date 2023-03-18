from transformers import AutoModelForMultipleChoice, AutoTokenizer, AutoModel
from global_vars import *
import torch
from torch import nn

model_saved = "bert_mlm22023-03-18-11-29-48.pt"
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
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


            for i in range(len(self.model.encoder.layer)):
                if(i < len(self.model.encoder.layer) + freeze_threshold):
                    for param in self.model.encoder.layer[i].parameters():
                        if(param.requires_grad):
                            param.requires_grad = False

    def forward(self, input_id, mask):

        _, o1 = self.model(input_ids= input_id, attention_mask=mask, return_dict=False)
        o2 = self.dropout(o1)
        o3 = self.linear(o2)
        fo = self.relu(o3)

        return fo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForClassification(freeze=True)
model.load_state_dict(torch.load(model_saved, map_location=torch.device('cpu')))
model = model.to(device)


from datasets import load_from_disk
import pandas as pd
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


ds = load_from_disk("BERT_TEST2023-03-18-11-29-48")
df = pd.DataFrame.from_dict({"posts": ds['posts'], "type": ds['type']})

df_test = Dataset(df)
test_dataloader = torch.utils.data.DataLoader(df_test, batch_size=4)

import numpy as np
from tqdm import tqdm
total_acc_test = 0


for test_input, test_label in tqdm(test_dataloader):
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc


print("Test Acc: " + str(total_acc_test / len(df_test)))