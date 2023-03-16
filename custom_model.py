from global_vars import *
from transformers import AutoModelForMultipleChoice, AutoTokenizer
from datasets import Dataset, load_from_disk
from tqdm import tqdm
import pandas as pd
import wandb
from wandb.keras import WandbMetricsLogger, WandbCallback
import pickle
import numpy as np

import tensorflow as tf
import os
from tensorflow.keras import Model # if only machine learning were this easy :P
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#import any other libraries you want here:
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, SpatialDropout1D, LSTM, Embedding, GlobalAveragePooling1D
from tensorflow.keras.activations import softmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping


dict = {
    "model_name": model_name,
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

df = pd.read_csv("converted_new_for_custom.csv")
df.info()

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 500
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['posts'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['posts'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(df['type']).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, name = "lstm_1", return_sequences=True))
model.add(LSTM(50, dropout=0.1, recurrent_dropout=0.1, name = "lstm_2", return_sequences=True))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='softmax', name = "dense"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


epochs = 2
batch_size = 256

model.summary()

history = model.fit(X_train, 
                    Y_train, 
                    epochs=epochs, 
                    batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001), 
                    WandbCallback()])
model.save(os.path.join(wandb.run.dir, "model.h5"))

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))