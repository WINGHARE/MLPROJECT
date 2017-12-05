"""
You dont need to care about the CNN, we just want to get usable features

"""# coding: utf-8

# In[24]:

#import pip

#package_name='shutil'
#pip.main(['install', package_name])

# In[1]:

import os
import re
import sys
import time
from optparse import OptionParser

import keras.backend as K
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import (LSTM, AveragePooling2D, Conv2D, Dense, Embedding,
                          Flatten,MaxPooling2D)
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (classification_report, fbeta_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

import feature_extract as f
import roc as roc

argvs = sys.argv

opts, args = {}, []

print(argvs)
print("##########")

def precision(y_true, y_pred):
    #Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1score(y_true, y_pred):
    r = recall(y_true, y_pred)
    p = precision(y_true, y_pred)
    return 2 * p * r / (p + r)


def decode_y(y, features=np.array([0, 1])):
    return np.dot(y, features).astype(int)



class TestCallback(Callback):
    def __init__(self, test_data1, test_data2):
        self.Xdata = test_data1
        self.Ydata = test_data2

    def on_epoch_end(self, epoch, logs={}):
        x = self.Xdata
        y = self.Ydata
        loss, acc, recall, pre, f1 = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def bulid_model(X_train,
                X_test,
                Y_train,
                Y_test,
                CID,
                fromfile='none'):
    model = Sequential()
    model.add(
        Conv2D(
            128,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding='same',
            activation='relu',
            input_shape=(X_train[0].shape[0],X_train[0].shape[1],1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(2, activation='softmax'))

    if (fromfile == 'none'):
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', recall, precision, f1score])
        print(model.summary())

        batch_size = 32

        checkpointer = ModelCheckpoint(
            filepath=os.path.join('tmp', 'weights_' + CID + '.hdf5'),
            verbose=1,
            save_best_only=True)

        model.fit(
            X_train,
            Y_train,
            epochs=15,
            batch_size=batch_size,
            verbose=1,
            validation_data=(X_test, Y_test),
            callbacks=[checkpointer])

        model.save_weights(
            filepath=os.path.join('tmp', 'weights_' + CID + '.hdf5'))
        return model

    else:
        filepath = os.path.join('tmp', fromfile)
        model.load_weights(filepath=filepath)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', recall, precision, f1score])
        print(model.summary())
        return model

    return model


def main():

    # CID is used to run on the department cluster

    CID = opts.cluster

    if (opts.load != 'none'): CID = opts.load

    X_train, X_test, Y_train, Y_test, enc = f.get_data()

    model = bulid_model(
        X_train, X_test, Y_train, Y_test, CID, fromfile=opts.load)

    newData = X_test.reshape(X_test.shape[0], 1, 100, 20)

    Y_score = model.predict_proba(X_test)

    roc.roc_plot(
        Y_test, Y_score, 2, filepath=os.path.join('figures', CID + 'roc.jpg'),title="CNN 2D AVG pool")

    Y_de = decode_y(Y_test, features=enc.active_features_)
    Y_pred = model.predict(X_test)
    Y_depred = decode_y(Y_pred, features=enc.active_features_)
    print(classification_report(Y_de, Y_depred))

    return


if __name__ == "__main__":

    plt.switch_backend('agg')
    mp.use('Agg')

    op = OptionParser()
    op.add_option(
        '-c',
        '--cluster',
        action='store',
        type='string',
        dest='cluster',
        help='indicate the clusterid')
    op.add_option(
        '-d',
        '--date',
        action='store',
        type='string',
        dest='date',
        help='indicate the date')
    op.add_option(
        '-l',
        '--load',
        default='none',
        action='store',
        type='string',
        dest='load',
        help='load weight form file')
    (opts, args) = op.parse_args()
    if len(args) > 0:
        op.print_help()
        op.error('Please input options instead of arguments.')
        exit(1)

    main()
