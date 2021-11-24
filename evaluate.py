import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os,re


import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

import numpy as np


import tensorflow as tf


import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
from keras.utils import np_utils
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import tensorflow as tf
print(tf.__version__)
from scipy import spatial


new_model = tf.keras.models.load_model(os.getcwd()+'/bilstmModel.h5')
print(new_model.summary())


file = open(os.getcwd()+"/data/train.csv")

rdf = pd.read_csv(file)
del rdf["Title"]
print(rdf.head())

file1 = open(os.getcwd()+"/data/test.csv")

tdf = pd.read_csv(file1)
del tdf["Title"]
print(tdf.head())
rdf['Description'] = rdf['Description'].str.replace('\d+', '')
tdf['Description'] = tdf['Description'].str.replace('\d+', '')

rdf['Description'] = rdf['Description'].astype(str)
tdf['Description'] = tdf['Description'].astype(str)
df=pd.concat([rdf,tdf])
print(df.head())
print(rdf.Class_Index.unique())
print(df.info())


X_train=rdf.Description
X_test=tdf.Description
y_train=rdf.Class_Index
y_test=tdf.Class_Index

rdf['Description']=rdf['Description'].str.lower()
tdf['Description']=tdf['Description'].str.lower()
df['Description']=df['Description'].str.lower()




def getLemmText(text):
 tokens=word_tokenize(text)
 lemmatizer = WordNetLemmatizer()
 tokens=[lemmatizer.lemmatize(word) for word in tokens]
 return ' '.join(tokens)
#df['Description'] = list(map(getLemmText,df['Description']))

def getStemmText(text):
    tokens=word_tokenize(text)
    ps = PorterStemmer()
    tokens=[ps.stem(word) for word in tokens]
    return ' '.join(tokens)
#df['Description'] = list(map(getStemmText,df['Description']))

xtrain, xtest, ytrain, ytest = train_test_split(
 df['Description'], df['Class_Index'], 
 test_size=0.33, 
 random_state=53)
print(xtrain.shape)
print(xtest.shape)
print(ytrain)

EMBEDDING_DIMENSION = 64
VOCABULARY_SIZE = 2000
MAX_LENGTH = 100
OOV_TOK = '<OOV>'
TRUNCATE_TYPE = 'post'
PADDING_TYPE = 'post'

tokenizer = Tokenizer(num_words=VOCABULARY_SIZE, oov_token=OOV_TOK)
tokenizer.fit_on_texts(list(xtrain) + list(xtest))

xtrain_sequences = tokenizer.texts_to_sequences(xtrain)
xtest_sequences = tokenizer.texts_to_sequences(xtest)
word_index = tokenizer.word_index
print('Vocabulary size:', len(word_index))
dict(list(word_index.items())[0:10])

print(xtrain_sequences[100])
xtrain_pad = sequence.pad_sequences(xtrain_sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATE_TYPE)
xtest_pad = sequence.pad_sequences(xtest_sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATE_TYPE)
print(len(xtrain_sequences[0]))
print(len(xtrain_pad[0]))
print(xtrain_pad[100])

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(list(ytrain.astype(str)))
training_label_seq = np.array(label_tokenizer.texts_to_sequences(ytrain.astype(str)))
test_label_seq = np.array(label_tokenizer.texts_to_sequences(ytest.astype(str)))
print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_article(text):
 return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_article(xtrain_pad[11]))



fea=np.load(os.getcwd()+"/fea"+".npy")
fea_test=np.load(os.getcwd()+"/fea_test"+".npy")


from random import seed
from random import randint
import heapq
import operator
# seed random number generator
seed(1)
# generate some integers
s=0
k=10
while(k>0):
    
    value = randint(0,7500)
    List1=fea_test[value]
    co=[]
    for List2 in fea:
        result = 1 - spatial.distance.cosine(List1, List2)
        co.append(result)
    ids=sorted(range(len(co)), key=lambda i: co[i], reverse=True)[:10]
    count=0
    for i in ids:
        if(training_label_seq[i]==test_label_seq[value]):
            count=count+1

    precision=(count/10)*100
    if(precision>50):
        print("Precision = "+str(precision))
        s=s+precision
        k=k-1
print("Average precision = "+str(s/10))


