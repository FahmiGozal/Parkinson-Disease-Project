import math
import numpy as np
import pandas as pd

def modify_columns(df):
    df_copy = df.copy()
    
    df_copy['X Displacement'] = 0
    df_copy['Y Displacement'] = 0
    df_copy['Distance'] = 0

    for i in range (1, df_copy.shape[0]):
        df_copy['X Displacement'][i] = abs(df_copy.X[i]-df_copy.X[i-1])
        df_copy['Y Displacement'][i] = abs(df_copy.Y[i]-df_copy.Y[i-1])
        df_copy['Distance'][i] = math.sqrt(df_copy['X Displacement'][i]^2 + df_copy['Y Displacement'][i]^2)

    df_copy.drop(['X', 'Y', 'Timestamp'], axis=1, inplace=True)
    
    return df_copy