import os
import re
import sys

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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

# nltk.download('words')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

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

    ## data pre processing
    ## require nltk data, if database missing, notification will be shown
    ## you can solve if by:
    ##
    ## import nltk
    ## nltk.download(words)

    data = pd.read_csv(os.path.join('data', '01.csv'), encoding="ISO-8859-1")
    text = data['text']
    sentiment = data['target']
    num_data = len(text) # get number of data
    print("number of tweets is " + str(num_data))

    text = text.apply(lambda x: x.lower())  #lowercase
    text = text.apply((
        lambda x: re.sub(r'[?|$|&|*|%|@|(|)|~]', '', x)))  #remove punctuations
    text = text.apply((
        lambda x: re.sub(r'[^a-zA-Z0-9 ]', '', x)))  #only numbers and alphabet
    text = text.apply(lambda x: removeStopWords(x))  #remove stopwords
    text = text.apply(lambda x: lemmat(x))  #lemmatize the sentence
    text = text.apply(lambda x: removeNonEnglish(x))  #remove

    ## enchode the original Y label tp one - hot form
    Y = [sentiment[i] for i in range(0, num_data)]
    Y = np.asarray(Y)
    ohenc = OneHotEncoder()
    Y2 = ohenc.fit_transform(Y.reshape(-1, 1)).toarray()


    ## We want to get X, X is a matrix with dimension:
    ## number of tweets * fixed lengh of articles * word vector size * 1 (1 channel)
    ## word vector can get from sklearn.feature_extraction.text.CountVectorizer
    ## By doing PCA on word cooccrurrence matrix : 
    ## https://stackoverflow.com/questions/35562789/word-word-co-occurrence-matrix

    texts = [text[i] for i in range(0, num_data)] # get text list

    count_model = CountVectorizer(ngram_range=(1,1)) # default unigram model
    co_mat = count_model.fit_transform(texts)

    # tokenizer = Tokenizer(split=' ')
    # testX = tokenizer.texts_to_sequences(texts)
    # print(len(testX))

    ## create a term dict with empty value
    term_dict = {}
    term_lst = count_model.get_feature_names()
    for t in term_lst:
        term_dict[t] = ""
    print(len(term_dict)) # 1885

    co_matc = (co_mat.T * co_mat) # this is co-occurrence matrix in sparse csr format
    co_matc.setdiag(0) # sometimes you want to fill same word cooccurence to "some value (1 or 0?)"
    co_matc = co_matc.todense()
    co_matc = np.asarray(co_matc)
    print(co_matc.shape)


    ## Dimension reduction by using PCA (how to decide )
    pca = PCA(n_components=0.9, copy=True) # if n_components is type 'float', it means keep 'f' percentage information
    pca.fit(co_matc) # train
    # print(pca.explained_variance_ratio_)
    # print(pca.explained_variance_)
    print(pca.n_components_)

    co_matc_new = pca.transform(co_matc) # get new matrix and original matrix does not change
    # print(co_matc_new[0:1, :])
    print(co_matc_new.shape)

    ##  Assign feature vector to each term
    for i in range(0, co_matc_new.shape[0]):
        term_dict[term_lst[i]] = co_matc_new[i] # np.array

    ## Assign each vectorized term to original text
    X = []
    for tx in texts:
        x = [] # each tweet is now a list of vectorized term
        for t in tx.split(' '):
            if t not in term_dict:
                pass
                # print(t + " not in term_dict")
            else:
                x.append(term_dict[t])
        X.append(x)
    print(len(X))

    ## padding X
    X = pad_sequences(X)
    print(X.shape)

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y2, test_size=0.4)

    
    '''
    ### Here is for non-pca term vector
    ### Replace co_matc_new with co_matc
    
    ##  Assign feature vector to each term
    for i in range(0, co_matc.shape[0]):
        term_dict[term_lst[i]] = co_matc[i] # np.array

    ## Assign each vectorized term to original text
    X = []
    for tx in texts:
        x = [] # each tweet is now a list of vectorized term
        for t in tx.split(' '):
            if t not in term_dict:
                pass
                # print(t + " not in term_dict")
            else:
                x.append(term_dict[t])
        X.append(x)
    print(len(X))
    '''

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
#     print(text_list)
#     return
#     X = tokenizer.texts_to_sequences(text_list)
#     print(X)
#     X = pad_sequences(X)
#     print(X)
#     return
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

    #get_data()
    # get_data2_emb()
    print("finish...")

    return


if __name__ == "__main__":
    main()


