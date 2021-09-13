import math
import numpy as np
import pandas as pd
from scipy.sparse.construct import random

#Local Library
import column_enhancement as ce
import array_for_model as afm
import parkinson_classifier as pclass

import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn import metrics #accuracy score, balanced_accuracy_score, confusion_matrix

#read the dataframe from the output of initial_data_compilation
df = pd.read_csv('train_model.csv')

#Read the Dataframe into a List before train_test_split
df2 = df.copy()
x = df2.iloc[:,:-1]
y = df2.iloc[:,-1]
df2.drop(['target'], axis=1, inplace=True)
x_array_of_list = df2.to_numpy()

# Trial - PCA
from sklearn.decomposition import PCA

#pca_model = PCA().fit(x)
#pca_x = pca_model.transform(x)

#Train_Test_Split (Within the Train Model)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state=3, stratify=y)

#This piece does not work
#No Need for train_test_split as it has been manually split
#x_train_alt, y_train_alt = x, y

#Create df for test dataset

df_test_control = pd.read_csv('test_control_model.csv')
df_test_parkinson = pd.read_csv('test_parkinson_model.csv')

x_test_control = df_test_control.iloc[:,:-1]
y_test_control = df_test_control.iloc[:,-1]

x_test_parkinson = df_test_parkinson.iloc[:,:-1]
y_test_parkinson = df_test_parkinson.iloc[:,-1]

# ML Model

    # Single ML Testing (Works)
'''
from sklearn.ensemble      import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)
pred = model.predict(x_test)


print(metrics.accuracy_score(y_test, pred)*100)
'''

#Prediction
with open('model\optimal_model.pkl', 'rb') as f:
    mp = pickle.load(f)
    pred = mp.predict(x_test_control)
    print(metrics.accuracy_score(y_test_control, pred)*100)

with open('model\optimal_model.pkl', 'rb') as f:
    mp = pickle.load(f)
    pred = mp.predict(x_test_parkinson)
    print(metrics.accuracy_score(y_test_parkinson, pred)*100)

with open('model\optimal_model.pkl', 'rb') as f:
    mp = pickle.load(f)
    pred = mp.predict(x_test)
    print(metrics.accuracy_score(y_test, pred)*100)
