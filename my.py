import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM

np.random.seed(0)

data = pd.read_csv('./data/1/Building5Electricity.csv')

data["Date"] = pd.to_datetime(data.Date, format='%d/%m/%Y')

data["Day_Type"] = data.Date.apply(lambda x: 1 if x.dayofweek > 4 else 0)

clean_data = data[["Date", "Day_Type", "0:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00", "4:30", "5:00", "5:30", "6:00", "6:30", "7:00", "7:30", "8:00", "8:30", 
                   "9:00", "9:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30",
                   "17:00", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30", "24:00:00",]]

trainData = clean_data[clean_data.Date <= pd.to_datetime("31/12/2013")]

trainDataU = trainData[["0:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00", "4:30", "5:00", "5:30", "6:00", "6:30", "7:00", "7:30", "8:00", "8:30", 
                   "9:00", "9:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30",
                   "17:00", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30", "24:00:00",]].values.reshape(13968, 1)

trainDataU[:, np.newaxis]

#print(trainDataU)
#print(trainDataU.shape)

trainDataD = trainData[["Day_Type"]].values

#trainDataT = clean_data[:0].values

testData = clean_data[pd.to_datetime("2013-12-31") < clean_data.Date]

testDataU = testData[["0:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00", "4:30", "5:00", "5:30", "6:00", "6:30", "7:00", "7:30", "8:00", "8:30", 
                   "9:00", "9:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30",
                   "17:00", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30", "24:00:00",]].values.reshape(3024, 1)

#print(testDataU.shape)

testDataD = testData[["Day_Type"]].values

XtrainDataU = trainDataU[: 10000]
YtrainDataU = trainDataU[48 : 10048]


XtestDataU = testDataU[49 : 97]
YtestDataU = testDataU[98 : 146]

#print(XtestDataU)
#print(YtestDataU)

def create_model(neurons):
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation='relu'))
    #model.add(Dropout(0.8))
    #model.add(Dense(1))
    model.add(Dense(1))
    model.compile(loss="mean_absolute_error", optimizer="nadam")
    return model

model = create_model(5)
model.fit(XtrainDataU, YtrainDataU, epochs=500, batch_size=500, verbose=2)

# Estimate model performance
trainScore = model.evaluate(XtrainDataU, YtrainDataU, verbose=0)
print('Train Score: %.10f MAE' % (trainScore))
testScore = model.evaluate(XtestDataU, YtestDataU, verbose=0)
print('Test Score: %.10f MAE' % (testScore))

# generate predictions for training
trainPredict = model.predict(XtrainDataU)
testPredict = model.predict(XtestDataU)

#print(testPredict)
#print(testPredict.shape)
plt.plot(testPredict)
plt.plot(YtestDataU)
#plt.plot(testPredictPlot)
plt.minorticks_on()
plt.show()

