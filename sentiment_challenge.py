#!/bin/python

import tflearn
from tflearn.data_utils import to_categorical
import pandas as pd
import numpy as np
from collections import Counter

""" 
Week three challenge of Intro to Deep Learning with Siraj. The challenge is to write a reccurernt network utilizing LSTM to classify video game reviews based on sentiment 

"""
# Need to vectorize the words
data = pd.read_csv('ign.csv')
data['title'] = data['title'].apply(lambda x: x.lower())

# Remove features which are not needed
# data = data.drop([data.columns[0],  'url', 'platform', 'score',  'genre', 'editors_choice', 'release_year', 'release_month', 'release_day'], axis=1)
len_train_data = int(len(data) * .9)
len_test_data = len(data) - len_train_data

def count_words(data):
    """ Returns a list of the words from the titles in order of number of occurences """
    word_counter = Counter()

    for i in data:
        for title in data['title']:
            for word in title.split(' '):
                word_counter[word.lower()] += 1
                
    word_counter = sorted(word_counter, key=lambda x: word_counter[x], reverse=True) # List of all word from the titles with most common word being first

    return [word for word in word_counter]
        
def title_to_vector(title, counts):
    """ """
    def get_num():
        """ """
        for word in title.split(' '):
            yield counts.index(word) 

    return [number for number in get_num()]

def make_word_vectors(data):
    """ """
    word_list = count_words(data)

    return [title_to_vector(title, word_list) for title in data['title']]
    
# Split data into testing and training sets
# train_x, train_y = data['score_phrase'][:len_train_data], data['title'][:len_train_data]
# test_x, test_y = data['score_phrase'][- len_test_data:], data['title'][-len_test_data:]

# # Convert labels to vectors

# train_y = to_categorical(train_y, nb_classes=10)
# test_y = to_categorical(test_y, nb_classes=10)

# net = tflearn.input_data([None, 100])
# net = tflearn.embedding(net, input_dim=len(data), output_dim=128)
# net = tflearn.lstm(net, 128, dropout=0.8)
# net = tf.fully_connected(net, 2, activation='softmax')
# net = tflearn.regression(net, optimizer='adam', learning_rate=.001, loss='categorical_crossentropy')

# model = tflearn.DNN(net, tensorboard_verbose=0)
# model.fit(train_x, train_y, vaidation_set=(test_x, test_y), show_metric=True, batch_size=32)




