
# MBTIPredictor

### Created by: [David Du](https://github.com/daduuu), [Seungmo Lee](https://github.com/543090lee), [Arthur Yang](https://github.com/bongjohn76), [Enoch Huang](https://github.com/ehuang3190)



This is a project that uses the MBTI personality test to predict the personality type of a user based on their social media posts. The dataset we use is the [MBTI 500](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset) dataset sourced from kaggle.
## Important Setups
Configurations for hyperparameters are set in [global_vars.py](https://github.com/daduuu/MBTIPredictor/blob/main/global_vars.py). If you want to run on CUDA device, make sure to edit that in the training files, specifically the line
```
device  =  torch.device("cuda:5"  if  torch.cuda.is_available() else  "cpu")
```
## How to Run
First download the [dataset](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset) and extract it. Rename it mbti_500.csv.

### LSTM Model Testing
To be done

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
```
python T5_test_file.py
```

