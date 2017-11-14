import os
import re

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

words = set(nltk.corpus.words.words())
stemmer = PorterStemmer()


def removeNonEnglish(s):
    sen = s
    return " ".join(
        w for w in nltk.wordpunct_tokenize(sen)
        if w.lower() in words or not w.isalpha())


def removeStopWords(ss):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(ss)
    sent = [w for w in tokens if not w in stop_words]
    sent2 = []
    for w in tokens:
        if w not in stop_words:
            sent2.append(w)
    return " ".join(sent2)


def lemmat(ss):
    lem = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(ss)
    sent = [lem.lemmatize(tokens[i], 'n') for i in range(len(tokens))]
    sent2 = [lem.lemmatize(sent[i], 'v') for i in range(len(sent))]
    return " ".join(sent2)


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def poter_tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return " ".join(stems)

def get_data():
    """Transform the text data to word features"""

    # data pre processing
    # require nltk data, if database missing, notification will be shown
    # you can solve if by:
    #
    # import nltk
    # nltk.download(database name)
    #
    data = pd.read_csv(os.path.join('data', '01.csv'), encoding="ISO-8859-1")
    text = data['text']
    sentiment = data['target']
    text = text.apply(lambda x: x.lower())  #lowercase
    text = text.apply((
        lambda x: re.sub(r'[?|$|&|*|%|@|(|)|~]', '', x)))  #remove punctuations
    text = text.apply((
        lambda x: re.sub(r'[^a-zA-Z0-9 ]', '', x)))  #only numbers and alphabet
    text = text.apply(lambda x: removeStopWords(x))  #remove stopwords
    text = text.apply(lambda x: lemmat(x))  #lemmatize the sentence
    text = text.apply(lambda x: removeNonEnglish(x))  #remove

    # enchode the original Y label tp one - hot form
    Y = [sentiment[i] for i in range(0, len(sentiment))]
    Y = np.asarray(Y)
    ohenc = OneHotEncoder()
    Y2 = ohenc.fit_transform(Y.reshape(-1, 1)).toarray()

    # We want to get X, X is a matrix with dimension:
    # number of tweets * fixed lengh of articles * word vector size * 1 (1 channel)
    # word vector can get from sklearn.feature_extraction.text.CountVectorizer
    # By doing PCA on word cooccrurrence matrix : 
    # https://stackoverflow.com/questions/35562789/word-word-co-occurrence-matrix


    X = []


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y2, test_size=0.4)

    # We return training set test set X_train and X_test. Y_train and Y_test
    # And the encoder to encoder Y
    return X_train, X_test, Y_train, Y_test, ohenc


# This is a messy example of feature extraction (embedding of CNN required) 
# 
# def get_data2_emb():
#     data = pd.read_csv(os.path.join('data', '01.csv'), encoding="ISO-8859-1")
#     text = data['text']
#     sentiment = data['target']
#     text = text.apply(lambda x: x.lower())  #lowercase
#     text = text.apply((
#         lambda x: re.sub(r'[?|$|&|*|%|@|(|)|~]', '', x)))  #remove punctuations
#     text = text.apply((
#         lambda x: re.sub(r'[^a-zA-Z0-9 ]', '', x)))  #only numbers and alphabet
#     text = text.apply(lambda x: removeStopWords(x))  #remove stopwords
#     text = text.apply(lambda x: lemmat(x))  #lemmatize the sentence

#     text = text.apply(lambda x: removeNonEnglish(x))  #remove

#     max_fatures = 2000
#     tokenizer = Tokenizer(num_words=max_fatures, split=' ')
#     text_list = [str(s.encode('ascii')) for s in text.values]
#     tokenizer.fit_on_texts(text_list)
#     X = tokenizer.texts_to_sequences(text_list)
#     X = pad_sequences(X)
#     X2 = tokenizer.texts_to_matrix(text_list, mode="tfidf")
#     X3 = [np.reshape(X2[i], (-1, 20)) for i in range(0, len(X2))]
#     X3 = np.asarray(X3)
#     X3 = X3.reshape(X3.shape[0], X3.shape[1], X3.shape[2], 1)

#     Y = [sentiment[i] for i in range(0, len(sentiment))]
#     Y = np.asarray(Y)

#     ohenc = OneHotEncoder()
#     Y2 = ohenc.fit_transform(Y.reshape(-1, 1)).toarray()

#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y2, test_size=0.4)
#     return X_train, X_test, Y_train, Y_test, ohenc

    

def main():

    

    return 


