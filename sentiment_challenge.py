#!/bin/python
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import pandas as pd
import numpy as np
from collections import Counter

""" 
Week three challenge of Intro to Deep Learning with Siraj. The challenge is to write a reccurernt network utilizing LSTM to classify video game reviews based on sentiment. 

Author: Ethan Henderson

Github @ethan626

"""

data = pd.read_csv('ign.csv')
data['title'] = data['title'].apply(lambda x: x.lower())

len_train_data = int(len(data) * .9)
len_test_data = len(data) - len_train_data

############## Function Definitions #################3

def count_words(data):
    """ Returns a list of the words from the titles in order of number of occurences """
    word_counter = Counter()

    for title in data:
        for word in title.split(' '):
            word_counter[word.lower()] += 1

    word_counter = sorted(word_counter, key=lambda x: word_counter[x], reverse=True) # List of all word from the titles with most common word being first

    return [word for word in word_counter]
        
def title_to_vector(title, counts):
    """ Changes the title of a video into an integer valued vector based on frequency. 1 is the most common word."""
    def get_num():
        """ Yields the numer corresponding to a word in the title. """
        for word in title.split(' '):
            yield counts.index(word) 

    return [number for number in get_num()]

def make_word_vectors(data):
    """ Call this function to return a list of the titles as integer valued vectors """
    word_list = count_words(data)

    return np.array([title_to_vector(title, word_list) for title in data])

def make_class_vectors(data):
    """ Returns a numpy array of the class encoded data """
    types = list(data.unique())

    def to_category():
       """ Yields a number representing the encoding of the datum """ 
       for score in data:
           yield types.index(score)

    return np.array([num_type for num_type in to_category()])

################## Data Preprocessing #######################

title_sequences= make_word_vectors(data['title']) # Numerically encode the video game titles 
score_vectors = make_class_vectors(data['score_phrase']) # Numerically encode the score_phrase the game received 
title_sequences = pad_sequences(title_sequences, maxlen=16, value=0) 

train_x, train_y =  title_sequences[:len_train_data],  score_vectors[:len_train_data] # Split data into training and testing sets 
test_x, test_y = title_sequences[-len_test_data:],  score_vectors[- len_test_data:] 

train_y = to_categorical(train_y, nb_classes=11)
test_y = to_categorical(test_y, nb_classes=11)

net = tflearn.input_data([None, 16]) # Build the network 
net = tflearn.embedding(net, input_dim=len(train_x), output_dim=11)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 11, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=.001, loss='categorical_crossentropy') # Using gradient descent 

model = tflearn.DNN(net, tensorboard_verbose=0)
model.load('sentiment_model.py')

if __name__ == '__main__':
    print('Training...')
    model.fit(train_x, train_y, validation_set=(test_x, test_y), show_metric=True) # Train the network
    model.save('sentiment_model.py')  

    score_phrases = data['score_phrase'].unique() # Validation 
    preds = model.predict(test_x)

    print('Prediction are...')  
    for score, game in zip(preds, data['title'][:-len_test_data]):
        print(game, ' - > ', score_phrases[np.argmax(score)])
    

