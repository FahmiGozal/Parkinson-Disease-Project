import math
import numpy as np
import pandas as pd
import column_enhancement as ce
import array_for_model as afm
import os

path_to_test_control = r'hw_dataset\test\control'
path_to_test_parkinson = r'hw_dataset\test\parkinson'
df_test_control = afm.create_specific_array(path_to_test_control, 0)
df_test_parkinson = afm.create_specific_array(path_to_test_parkinson, 0)

df_test_control.to_csv('test_control_model.csv', index=False, header = True)
df_test_parkinson.to_csv('test_parkinson_model.csv', index=False, header = True)




