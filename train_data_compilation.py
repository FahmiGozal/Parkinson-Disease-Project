import math
import numpy as np
import pandas as pd
import column_enhancement as ce
import array_for_model as afm
import os

path ='hw_dataset/train'
df_save = afm.create_the_array(path)

df_save.to_csv('train_model.csv', index=False, header = True)




