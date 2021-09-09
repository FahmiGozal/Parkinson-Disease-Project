import math
import numpy as np
import pandas as pd
import column_enhancement as ce
import os

def classification_pairing(df, clf, seq_length=6):
    x=[]
    y=[]
    for i in range(0, (df.shape[0]- seq_length +1), seq_length+1):
        seq= np.zeros((seq_length, df.shape[1]))
        for j in range(seq_length):
            seq[j]= df.values[i+j]
        x.append(seq.flatten())
        y.append(clf)
    return x, y

def create_the_array():
    path ='hw_dataset'

    #Combine all data into a single array to feed to ML model
    #Empty x and y as placeholder for final use
    x = []
    y = []

    columns = ['X', 'Y', 'Z', 'Pressure', 'Grip Angle', 'Timestamp', 'Test ID' ]
    
    for i, folder1 in enumerate(os.listdir(path)):
        for f in os.listdir(path + '/' + folder1):
            df = pd.read_csv(path + '/' + folder1 + '/' + f, sep = ';', names = columns)
            
            #use the enhanced data
            dfe = ce.modify_columns(df)

            x_temp, y_temp = classification_pairing(dfe, i)
            x.append(x_temp)
            y.append(y_temp)
    
    array_of_x = np.concatenate(x, axis = 0)
    array_of_y = np.concatenate(y, axis = 0)

    df_save = np.concatenate(array_of_x, axis = 0)
    df_save['target'] = array_of_y

    return df_save