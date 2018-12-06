# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:53:44 2018

@author: liuzh
"""

import tarfile
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from string import digits, punctuation
from sklearn.neighbors import KDTree
from scipy.stats import mode

import re
import vectorize 
import train_model 

#read the news and their labels from a given tar file
def LoadData(fname):
    tar = tarfile.open(fname, 'r:gz', errors = 'ignore')
    all_news = []
    labels = []
    for tarinfo in tar:
        if tarinfo.isreg():
            content = tar.extractfile(tarinfo).read()
            all_news.append(content)
            label = tarinfo.name.replace('.','/').split('/')
            labels.append(label[1])
    tar.close()
    return all_news, labels

def split_data(X, y, test_size, random_state = 12):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = random_state)
    return X_train, X_test, y_train, y_test

def clean_data(sample):
    all_f = []
    for i in sample:
        text = i.decode('utf-8',errors = 'ignore').lower()
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"\'m", " am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'s"," is",text)
        text = text.translate(str.maketrans(dict.fromkeys(punctuation+digits, " ")))
        text = ' '.join(text.split())
        all_f.append(text)
    return all_f

def pred(model, x_test, y_test):
    prediction = np.argmax(model.predict(x_test),axis=1)
    diff = 0
    for i in range(len(prediction)):
        if prediction[i] != y_test[i]:
            diff += 1
    acc = (y_test.shape[0] - diff)/y_test.shape[0]
    return acc

#Use KNN to predict the labels of test set
def knn(x_train, y_train, x_test, y_test, k):
    tree = KDTree(x_train.toarray(), leaf_size = 100)
    acc = np.zeros((len(k),1))
    for x in range(len(k)):
        pred_list = np.zeros(y_test.shape)
        t1 = time.time()
        for i in range(x_test.shape[0]): 
            _, index = tree.query(x_test[i].toarray(), k = k[x])
            pred_labels = y_train[index]
            most_freq = int(mode(pred_labels[0])[0])
            pred_list[i] = most_freq
        t2 = time.time()
        diff = np.array(pred_list) - np.array(y_test)
        acc[x] = sum(diff==0)/y_test.shape[0]
        print('Accuracy of KNN with k = %s is %s, it costs %s seconds' % (k[x], acc[x], t2-t1))
    return acc
   
#Load Data
all_news, labels = LoadData('20news-18828.tar.gz')
labels_num, labels_uniques = pd.factorize(labels)

#Clean Data
clean_all_news = clean_data(all_news)

#Split Data
[X_train, X_test, y_train, y_test] = split_data(clean_all_news, labels_num, 0.2)
[X_train, X_val, y_train, y_val] = split_data(X_train, y_train , 0.2, random_state = 15)

#KNN
[x_train_knn, x_val_knn, x_test_knn] = vectorize.ngram_vectorize(X_train, y_train, X_val, X_test,TOP_K=2000)

acc_knn = knn(x_train_knn, y_train, x_test_knn,y_test,[11,21,31])

#MLP
[x_train_mlp, x_val_mlp, x_test_mlp] = vectorize.ngram_vectorize(X_train, y_train, X_val, X_test)
data_mlp = ((x_train_mlp, y_train),(x_val_mlp, y_val))
model_mlp = train_model.train_mlp_model(data_mlp)
acc_mlp = pred(model_mlp, x_test_mlp, y_test)
print('The accuracy of MLP on test data is %s' % acc_mlp)

#CNN
[x_train_cnn, x_val_cnn, x_test_cnn, word_index] = vectorize.sequence_vectorize(X_train, X_val, X_test)
data_cnn = ((x_train_cnn, y_train),(x_val_cnn, y_val))

model_cnn = train_model.train_sequence_model(data=data_cnn,word_index=word_index,
                             embedding_data_dir='E:\SIT\CPE646\Final\CNN')

acc_cnn = pred(model_cnn, x_test_cnn, y_test)
print('The accuracy of pre-trained CNN on test data is %s' % acc_cnn)
