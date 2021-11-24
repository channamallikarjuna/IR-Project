import requests
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler

import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

model = Sequential()
forward_layer = LSTM(10, return_sequences=True)
backward_layer = LSTM(10, activation='relu', return_sequences=True,go_backwards=True)
model.add(Bidirectional(forward_layer, backward_layer=backward_layer,input_shape=(5, 10)))
model.add(Dense(5))
model.add(Flatten())
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print(model.summary())

import csv
import os
from transformers import XLNetTokenizer, XLNetModel
file = open(os.getcwd()+"/data/train.csv")
csvreader = csv.reader(file)
header = next(csvreader)
print(header)
rows = []
for row in csvreader:
    rows.append(row)
print(len(rows))
print(rows[0])
file.close()

