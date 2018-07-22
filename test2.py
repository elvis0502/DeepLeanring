import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
#from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM

np.random.seed(0)

data = pd.read_csv('./data/1/Building5Electricity.csv')

data["Date"] = pd.to_datetime(data.Date, format='%d/%m/%Y')

data["Day_Type"] = data.Date.apply(lambda x: 1 if x.dayofweek > 4 else 0)

clean_data = data[["Date", "Day_Type"]
