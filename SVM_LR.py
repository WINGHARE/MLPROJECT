import os
import re
import sys
import time

import matplotlib.pyplot as plt
import matplotlib as mp

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import (
    CountVectorizer, HashingVectorizer, TfidfTransformer)
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import (ShuffleSplit, learning_curve,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

import roc as roc
import feature_extract as f

plt.switch_backend('agg')
mp.use('Agg')

train_data, test_data, train_target, test_target, enc = f.get_data()

train_data = train_data.reshape(train_data.shape[0],-1)
test_data = test_data.reshape(test_data.shape[0],-1)
train_target = np.argmax(train_target,axis=1)
test_target= np.argmax(test_target,axis=1)

########

clf2 = Pipeline([

    ('clf', SGDClassifier(
        loss='log',
        random_state=42,
        shuffle=True,
        alpha=0.0001 * 0.75,
        penalty='l1',
        max_iter=20)),
])

_ = clf2.fit(train_data, train_target)
predicted = clf2.predict(test_data)
proba = clf2.predict_proba(test_data)
ohenc = OneHotEncoder()
Y2 = ohenc.fit_transform(test_target.reshape(-1, 1)).toarray()
roc.roc_plot(
    Y2, proba, 2, filepath=os.path.join('figures', 'tradSGD' + 'roc.svg'),title='Logistic',fmt='svg')

print(metrics.classification_report(
    test_target,
    predicted,
))

print(accuracy_score(test_target, predicted))
"""

try SVC

"""

clf5 = Pipeline([

    ('clf', SVC(
        C=1.25,
        cache_size=200,
        class_weight=None,
        coef0=0.0,
        probability=True,
        decision_function_shape=None,
        degree=1,
        gamma='auto',
        kernel='rbf',
        max_iter=-1,
        random_state=None,
        shrinking=True,
        tol=0.001,
        verbose=False)),
])

_ = clf5.fit(train_data, train_target)
predicted = clf5.predict(test_data)
proba = clf5.predict_proba(test_data)
print(metrics.classification_report(
    test_target,
    predicted,
))

roc.roc_plot(Y2, proba, 2, filepath=os.path.join('figures', 'SVC' + 'roc.svg'),title='SVM',fmt='svg')
print(accuracy_score(test_target, predicted))
