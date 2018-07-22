import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM

np.random.seed(0)

#Load data
data = pd.read_csv('./data/1/Building5Electricity.csv')
data["Date"] = pd.to_datetime(data.Date, format='%d/%m/%Y')
data["Day_Type"] = data.Date.apply(lambda x: 1 if x.dayofweek > 4 else 0)
trainData = data[data.Date <= pd.to_datetime("29/12/2013")]
dateTypeTrain = trainData[["Day_Type"]].values
dateTypeTrain = dateTypeTrain.reshape(1, dateTypeTrain.shape[0])
trainData = trainData[["0:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00", "4:30", "5:00", "5:30", "6:00", "6:30", "7:00", "7:30", "8:00", "8:30", 
                   "9:00", "9:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30",
                   "17:00", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30", "24:00:00",]]

testData = data[pd.to_datetime("2013-12-31") < data.Date]
testData = testData[testData.Date < pd.to_datetime("2014-1-22")]
dateTypeTest = testData[["Day_Type"]].values
dateTypeTest = dateTypeTest.reshape(1, dateTypeTest.shape[0])
testData = testData[["0:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00", "4:30", "5:00", "5:30", "6:00", "6:30", "7:00", "7:30", "8:00", "8:30", 
                   "9:00", "9:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30",
                   "17:00", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30", "24:00:00",]]


#Create training and testing datasets
def create_dataset(dataset, trainDatashape_1, trainDatashape_0, dateTypeshape_1, dateType):
    #Add and convert time information to an array
    timeT, timeF, dayF = [], [], []
    for i in range(trainDatashape_1):
        time = np.array([i])
        timeT = np.concatenate((timeT, time), axis=0)
    for i in range(trainDatashape_0):
        timeF = np.concatenate((timeF, timeT), axis=0)
    timeF = timeF.reshape(trainDatashape_0*trainDatashape_1, 1)
    #Add and convert week/weekend to an array
    for i in range(dateTypeshape_1):
        if dateType.item(i) == 0:
            for i in range(trainDatashape_1):
                day = np.array([0])
                dayF = np.concatenate((dayF, day), axis=0)
        else:
            for i in range(trainDatashape_1):
                day = np.array([1])
                dayF = np.concatenate((dayF, day), axis=0)
    dayF = dayF.reshape(dayF.shape[0], 1)
    #Convert trainData to an array
    dataset = np.asarray(dataset).reshape(trainDatashape_0*trainDatashape_1, 1)
    #Merge dayF, time, and usage to an array
    createData = np.concatenate((timeF, dataset), axis=1)
    createData = np.concatenate((dayF, createData), axis=1)
    return createData

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

trainData = create_dataset(trainData, trainData.shape[1], trainData.shape[0], dateTypeTrain.shape[1], dateTypeTrain)
testData = create_dataset(testData, testData.shape[1], testData.shape[0], dateTypeTest.shape[1], dateTypeTest)

tnData = series_to_supervised(trainData, 1, 1).values
ttData = series_to_supervised(testData, 1, 1).values

trainX, trainY = tnData[:, :-1], tnData[:, -1]
testX, testY = ttData[:, :-1], ttData[:, -1]

trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))

#Create model
model = Sequential()
model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(trainX, trainY, epochs=100, batch_size=72, validation_data=(testX, testY), verbose=2, shuffle=False)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.10f MSE' % (trainScore))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.10f MSE' % (testScore))

# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)



# plot history
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')

plt.plot(testY)
plt.plot(testPredict)
plt.legend()
plt.show()



















'''
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
plt.show()'''
