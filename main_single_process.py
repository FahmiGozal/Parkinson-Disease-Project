import math
import numpy as np
import pandas as pd
from scipy.sparse.construct import random
import column_enhancement as ce
import array_for_model as afm
import os

from sklearn.model_selection import train_test_split
from sklearn import metrics #accuracy score, balanced_accuracy_score, confusion_matrix

#read the dataframe from the output of initial_data_compilation
df = pd.read_csv('model_array.csv')

#Read the Dataframe into a List before train_test_split
df2 = df.copy()
x = df2.iloc[:,:-1]
y = df2.iloc[:,-1]
df2.drop(['target'], axis=1, inplace=True)
x_array_of_list = df2.to_numpy()

#train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_array_of_list, y, random_state=42, train_size=0.75, stratify=y)

# ML Model

from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import RandomForestClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier

#model = TimeSeriesForestClassifier() #Needs to be univariate (X.shape[1] == 1)
model = RandomForestClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(metrics.accuracy_score(y_test,y_pred)*100)
print(metrics.balanced_accuracy_score(y_test,y_pred)*100)
print(metrics.confusion_matrix(y_test,y_pred))






