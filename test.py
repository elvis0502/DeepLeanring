import numpy as np
import pandas as pd

a = np.array([[1]])
b = np.array([[3]])
c = np.concatenate((a,b), axis=0)
print (c)