import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

import matplotlib.pyplot as plt
'exec(%matplotlib inline)'
from subprocess import check_output

input_path = os.path.dirname(os.path.realpath(__file__))+'/input/Sentiment_GOP.csv'
data = pd.read_csv(input_path, encoding = "ISO-8859-1")
data = data[['text','sentiment']]

# Splitting the dataset into train and test set
train, test = train_test_split(data,test_size = 0.1)
# Removing neutral sentiments
train = train[train.sentiment != "Neutral"]

train_pos = train[ train['sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neg = train[ train['sentiment'] == 'Negative']
train_neg = train_neg['text']

tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    print(row.text.split())
    print(words_filtered)
    print("\n")
    # words_cleaned = [word for word in words_filtered
    #     if 'http' not in word
    #     and not word.startswith('@')
    #     and not word.startswith('#')
    #     and word != 'RT']
    # words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    # tweets.append((words_without_stopwords, row.sentiment))
    break


test_pos = test[ test['sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[ test['sentiment'] == 'Negative']
test_neg = test_neg['text']