'''
Created on May 24, 2024
Author: @G_Sansigolo
'''

# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

path_dir = os.path.dirname(__file__)

# read csv file
df = pd.read_csv(os.path.join(path_dir, 'train.csv'))

# shuffle data
df = shuffle(df)

print(df.head())

X_train = df["sentence"].fillna("fillna").values
y_train = df[["BookRestaurant", "GetWeather", "PlayMusic", "RateBook"]].values

print(X_train.shape, y_train.shape)

#Preprocessing

text = X_train

Tokenizer = Tokenizer()

# text preprocessing
Tokenizer.fit_on_texts(text) 
Tokenizer_vocab_size = len(Tokenizer.word_index) + 1

print(Tokenizer_vocab_size)

samples = 2500

X_train = X_train[samples:] 
y_train = y_train[samples:] 

X_val = X_train[:samples] 
y_val = y_train[:samples] 

X_train_encoded_words = Tokenizer.texts_to_sequences(X_train)
X_val_encoded_words = Tokenizer.texts_to_sequences(X_val)

X_train_encoded_padded_words = sequence.pad_sequences(X_train_encoded_words, maxlen = 100)
X_val_encoded_padded_words = sequence.pad_sequences(X_val_encoded_words, maxlen = 100)

print(X_train_encoded_padded_words.shape, X_val_encoded_padded_words.shape)

print(X_val_encoded_padded_words, X_train_encoded_padded_words)

#Build and Train the Model

model = Sequential()

model.add(Embedding(Tokenizer_vocab_size, 32, input_length = 100)) 

model.add(LSTM(10))
model.add(Dropout(0.5))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(200, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

print(model.summary())

Nadam = tf.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=Nadam, metrics=['accuracy'])

history = model.fit(X_train_encoded_padded_words,y_train, epochs = 3, batch_size=32, verbose=1, validation_data=(X_val_encoded_padded_words, y_val))

# save the model
model.save("Intent_Classification.h5")

#plot

#accuracy
from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

#Predictions

def predict(text):
    sentence = text
    tokens = Tokenizer.texts_to_sequences([text])
    tokens = pad_sequences(tokens, maxlen = 100)
    prediction = model.predict(np.array(tokens))
    pred = np.argmax(prediction)
    classes = ['BookRestaurant','GetWeather','PlayMusic','RateBook']
    result = classes[pred]
    return result

print(predict("is it raining ?"))

print(predict("i would like to book a table at hotel Orion for 29th june"))