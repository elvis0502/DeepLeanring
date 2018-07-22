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

trainData = trainData[["Day_Type", "0:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00", "4:30", "5:00", "5:30", "6:00", "6:30", "7:00", "7:30", "8:00", "8:30", 
                   "9:00", "9:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30",
                   "17:00", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30", "24:00:00",]]

testData = clean_data[pd.to_datetime("2013-12-31") < clean_data.Date]

testData = testData[testData.Date < pd.to_datetime("2014-1-22")]

testData = testData[["Day_Type", "0:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00", "4:30", "5:00", "5:30", "6:00", "6:30", "7:00", "7:30", "8:00", "8:30", 
                   "9:00", "9:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30",
                   "17:00", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30", "24:00:00",]]

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:i+look_back])
        dataY.append(dataset[i+look_back])
    return np.asarray(dataX), np.asarray(dataY)

look_back = 7

trainData = np.asarray(trainData)

trainX, trainY = create_dataset(trainData, look_back)


testData = np.asarray(testData)

testX, testY = create_dataset(testData, look_back)


print(testX.shape)
print(testY.shape)

model = Sequential()
model.add(LSTM(look_back, input_shape=(look_back, 49), activation='relu'))
model.add(Dense(49))
model.add(Dense(49))
model.compile(loss="mean_squared_error", optimizer="adam")
history = model.fit(trainX, trainY, epochs=8000, batch_size=500, validation_data=(testX, testY), verbose=2, shuffle=False)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.10f MSE' % (trainScore))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.10f MSE' % (testScore))

# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

testPredict = np.delete(testPredict, 0, axis=1)
testY = np.delete(testY, 0, axis=1)
testPredict = testPredict.reshape((testPredict.shape[0]*testPredict.shape[1]), 1)
testY = testY.reshape((testY.shape[0]*testY.shape[1]), 1)

#print(testPredict)
#print(testPredict.shape)
plt.plot(testY)
plt.plot(testPredict)

#plt.plot(testPredictPlot)
plt.minorticks_on()
plt.show()




