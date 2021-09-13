import math
import numpy as np
import pandas as pd
import column_enhancement as ce
import os

def classification_pairing(df, clf, seq_length=100):
    x=[]
    y=[]
    for i in range(0, (df.shape[0]- seq_length +1), seq_length+1):
        seq= np.zeros((seq_length, df.shape[1]))
        for j in range(seq_length):
            seq[j]= df.values[i+j]
        x.append(seq.flatten())
        #conditional if distance average is too big
        y.append(clf) # To be Changed (with distance.mean() or distance.var())
    return x, y

#Classification Based on Test ID

def classification_pairing_2(df, clf, seq_length=100):
    x=[]
    y=[]
    for test_id in df['Test ID']:
        df_temp = df[df['Test ID']==test_id]
        seq= np.zeros((df_temp.shape[0], df_temp.shape[1]))
        for j in range(df_temp.shape[0]):
            seq[j]=df_temp.values[j]
        x.append(seq.flatten())
        y.append(clf)

    return x, y

def create_the_array(path):

    #Combine all data into a single array to feed to ML model
    #Empty x and y as placeholder for final use
    x = []
    y = []

    columns = ['X', 'Y', 'Z', 'Pressure', 'Grip Angle', 'Timestamp', 'Test ID' ]
    
    #z = 1
    for i, folder1 in enumerate(os.listdir(path)):
        for f in os.listdir(path + '/' + folder1):
            df = pd.read_csv(path + '/' + folder1 + '/' + f, sep = ';', names = columns)
            
            #use the enhanced data
            new_columns = ['Z','Pressure','Grip Angle','Test ID','X Displacement','Y Displacement','Distance', 'Stable']
            dfe = ce.modify_columns(df)

            #Saving Individual DataFrame (For Checking Purpose)
            #path2 = 'individual_dataframe'
            #dfe.to_csv(path2 + '/' + folder1 + '/' + str(z) + '.csv',index= False, header =True)
            #z += 1

            #New Classification on Distance Stability
            #if df_copy['Distance'].mean() < 2:
                #df_copy['Stable'] = 0
            #else:
                #df_copy['Stable'] = 1

            x_temp, y_temp = classification_pairing(dfe, i)
            x.append(x_temp)
            y.append(y_temp)
    
    array_of_x = np.concatenate(x, axis = 0)
    array_of_y = np.concatenate(y, axis = 0)

    df_save = pd.DataFrame(array_of_x)
    df_save['target'] = array_of_y

    return df_save

def create_specific_array(path, i):

    #Combine all data into a single array to feed to ML model
    #Empty x and y as placeholder for final use
    x = []
    y = []

    columns = ['X', 'Y', 'Z', 'Pressure', 'Grip Angle', 'Timestamp', 'Test ID' ]
    
    #z = 1
    for f in os.listdir(path):
        df = pd.read_csv(path + '/' + f, sep = ';', names = columns)
            
        #use the enhanced data
        new_columns = ['Z','Pressure','Grip Angle','Test ID','X Displacement','Y Displacement','Distance', 'Stable']
        dfe = ce.modify_columns(df)

        #Saving Individual DataFrame (For Checking Purpose)
        #path2 = 'individual_dataframe'
        #dfe.to_csv(path2 + '/' + folder1 + '/' + str(z) + '.csv',index= False, header =True)
        #z += 1

        #New Classification on Distance Stability
        #if df_copy['Distance'].mean() < 2:
            #df_copy['Stable'] = 0
        #else:
            #df_copy['Stable'] = 1

        x_temp, y_temp = classification_pairing(dfe, i)
        x.append(x_temp)
        y.append(y_temp)
    
    array_of_x = np.concatenate(x, axis = 0)
    array_of_y = np.concatenate(y, axis = 0)

    df_save = pd.DataFrame(array_of_x)
    df_save['target'] = array_of_y

    return df_save