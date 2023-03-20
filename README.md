
# MBTIPredictor

### Created by: [David Du](https://github.com/daduuu), [Seungmo Lee](https://github.com/543090lee), [Arthur Yang](https://github.com/bongjohn76), [Enoch Huang](https://github.com/ehuang3190)



This is a project that uses models such as RoBERTa, T5, and LSTM to predict the personality type of a user based on their social media posts. The dataset we use is the [MBTI 500](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset) dataset sourced from kaggle.
## Important Setups
Configurations for hyperparameters are set in [global_vars.py](https://github.com/daduuu/MBTIPredictor/blob/main/global_vars.py). If you want to run on CUDA device, make sure to edit that in the training files, specifically the line
```
device  =  torch.device("cuda:5"  if  torch.cuda.is_available() else  "cpu")
```
## How to Run
First download the [dataset](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset) and extract it. Rename it mbti_500.csv.

### wandb setup
Create an account at [wandb](https://wandb.ai/). Install wandb python module and login with API key into terminal by using 
```
wandb login
```
Or on windows
```
python -m wandb login
```
Create a team and project to log the graphs and update the lines
```
wandb.init(
project="$PROJECT_NAME",
config=dict,
entity="$TEAM_NAME"
)
```

### LSTM Model Testing
First we have to generate the converted CSV
```
python convertedCSV_for_custom.py
```
Now you can run the custom_model.ipynb file, which tokenizes and trains the data.
Our finished training model is stored as my_model.zip

### RoBERTa Model Testing
First we have to generate the converted CSV. 
```bash
python convertedCSV.py
```
Next, we can run the training file. This takes care of tokenization for us.
```
python train_bert_for_sequence_classification.py
```
This also generates a testing file to calculate our testing accuracies.
```
BERT_TEST$TIME_STAMP
```
Now we can calculate the testing accuracy. Make sure to update these lines with the correct timestamp
```
model_saved = "bert_mlm22023-03-18-11-29-48.pt"
ds = load_from_disk("BERT_TEST2023-03-18-11-29-48")
```
Now we can run the following

```
python test_bert_for_sequence_classification.py
```

### T5 Model Testing
First we have to generate the converted CSV
```
python convertedCSV_for_T5.py
```
Next, we tokenize the data
```
python tokenise_data_for_T5.py
```
This generates a new folder 
```
dataset$TIME_STAMP
```
That has our train, test, and validation inputs and labels. Now we can run our training file.
```
python T5_training_file.py
```
Now we can run our test script. Make sure to update these lines with the correct timestamp.
```
model_saved = "t52023-03-18-13-13-00.pt"
time = "2023-03-18-04-46-28"
```
Now we can run the following
```
python T5_test_file.py
```

### Link to fine-tuned models
Here are the [links](https://mega.nz/file/aR5mhZ4L#oCL-LRS_NqDY32udHqISnNy-cFS2E3CRL2utPgVZmi4) to the fine-tuned RoBERTa and T5 models and the corresponding datasets that we trained and used to calculate our test accuracy. 

Careful! This file is around ~1.3 GB
