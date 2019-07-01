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

###

data = pd.read_csv("C:/Users/cathl/OneDrive/Documents/GitHub/BU-Grow-Starter-Project/input/Sentiment_GOP.csv", encoding = "ISO-8859-1")
data = data[['text','sentiment']]

# Splitting the dataset into train and test set
train, test = train_test_split(data,test_size = 0.1)
# Removing neutral sentiments
train = train[train.sentiment != "Neutral"]

train_pos = train[ train['sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neg = train[ train['sentiment'] == 'Negative']
train_neg = train_neg['text']

###

tweets = []
stopwords_set = set(stopwords.words("english")) #stopwords are words like is, are, that

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in nltk.word_tokenize(row.text) if len(e) >= 3] #gets rid of words that are shorter than 3
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']   #gets rid of words with http, @, #, or RT
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]  #gets rid of stopwords
    tweets.append((words_without_stopwords, row.sentiment))
    

test_pos = test[ test['sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[ test['sentiment'] == 'Negative']
test_neg = test_neg['text']

###

# Extracting word features

#returns list of all words
def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    return all 

#
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)   
    #FreqDist is a class that is like dictionary with word as key and # of times it appears as value

    features = wordlist.keys()
    return features

w_features = get_word_features(get_words_in_tweets(tweets)) #All unique words in all tweets

def extract_features(document):
    document_words = set(document)  #document was list of words, now list of unique words
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words) #adds word from all unique words into features if that word is in document
    return features


###

# Training the Naive Bayes classifier
training_set = nltk.classify.apply_features(extract_features,tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

###

def accuracyScore(correct, total):
    return correct/total      #How likely that u answer right

def sensitivityScore(truePos, totalPos):
    return truePos/totalPos    #How likely detect positive when is positive
    
def specificityScore(trueNeg, totalNeg):
    return trueNeg/totalNeg   #How likely detect negative when is negative
    
def precisionScore(truePos, guessedPos):
    return truePos/guessedPos #How likely is positive when detect positive

def f1Score(sensitivity, precision):
    return 2*sensitivity*precision/(sensitivity+precision)  
    #weighted avg of recall and precision, useful if harm of falsePos and falseNeg differs

###

trueNeg = 0
truePos = 0
falseNeg = 0
falsePos = 0

#Detecting Negatives
wrongNeg = []
count = 0
for obj in test_neg: 
    processedObj = extract_features(nltk.word_tokenize(obj))
    res =  classifier.classify(processedObj)
    if(res == 'Negative'): 
        trueNeg+=1
    else:
        falsePos+=1
        if(count<20):
            count+=1
            wrongNegs.append(processedObj)
        
#Detecting Positives
wrongPos = []
count = 0
for obj in test_pos:
    processedObj = extract_features(nltk.word_tokenize(obj))
    res =  classifier.classify(processedObj)
    if(res == 'Positive'): 
        truePos+=1
    else:
        falseNeg+=1
        if(count<20):
            count+=1
            wrongNegs.append(processedObj)

###

print(str(truePos)+" "+ str(falseNeg))       
print(str(falsePos)+ " "+ str(trueNeg))  

accuracy = accuracyScore(truePos+trueNeg, truePos+trueNeg+falsePos+falseNeg)
sensitivity = sensitivityScore(truePos,truePos+falseNeg)
specificity = specificityScore(trueNeg, trueNeg+falsePos)
precision = precisionScore(truePos, truePos+falsePos)
f1 = f1Score(sensitivity, precision)

print(accuracy)
print(sensitivity)
print(specificity)
print(precision)
print(f1)

###

for obj in wrongNeg:
    print(obj)
    print("\n")

###

for obj in wrongPos:
    print(obj)
    print("\n")


