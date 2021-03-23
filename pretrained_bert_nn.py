from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import keras
import sklearn
from sklearn.linear_model import SGDClassifier
from keras.models import Sequential
from keras.layers import Dense

BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 10

def data_generator(loc):
    df = pd.DataFrame([eval(i) for i in open(loc).readlines()]).iloc[:500000]
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    for i in range(len(df['text']) // BATCH_SIZE - 1):
        texts = list(df['text'].values[i * BATCH_SIZE: (i + 1) * BATCH_SIZE])
        encoded_texts = model.encode(texts)
        ratings = df['stars'].values[i * BATCH_SIZE: (i + 1) * BATCH_SIZE].astype(int) - 1
        ratings = keras.utils.to_categorical(ratings, NUM_CLASSES)
        yield encoded_texts.astype('float32'), ratings.astype('float32')

def test_data_generator(loc):
    df = pd.DataFrame([eval(i) for i in open(loc).readlines()]).iloc[500000:]
    print("testing length:", len(df))
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    texts = list(df['text'])
    encoded_texts = model.encode(texts)
    ratings = df['stars'].astype(int) 
    return encoded_texts.astype('float32'), ratings.astype('float32')
model = Sequential()
model.add(Dense(256, input_shape = (768, ), activation = 'tanh'))
model.add(Dense(128, activation = 'tanh'))
model.add(Dense(64, activation = 'tanh'))
model.add(Dense(32, activation = 'tanh'))
model.add(Dense(5, activation = 'softmax'))
model.summary()

model.compile('Adam', 'categorical_crossentropy', metrics = ['accuracy', 'mean_absolute_error'])
model.fit(data_generator('yelp_review_training_dataset.jsonl'), steps_per_epoch = 500000 / BATCH_SIZE, epochs = EPOCHS)
"""
model = SGDClassifier()
for i in range(5):
    print("Epoch", i)
    training_data = iter(data_generator('yelp_review_training_dataset.jsonl'))
    for X, y in training_data:
        model.partial_fit(X, y, classes = [1, 2, 3, 4, 5])
X, y = test_data_generator('yelp_review_training_dataset.jsonl')
print(np.mean(np.abs(model.predict(X) - y)))
"""
