import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM

np.random.seed(0)

def create_model(inputs, neurons, dropout =0.1):
    model = Sequential()
    model.add(Dense(100, input_dim=24, activation='relu'))
    model.add(Dropout(dropout))
    #model.add(Dense(1))
    model.add(Dense(1))
    model.compile(loss="mean_absolute_error", optimizer="nadam")
    return model

df = pd.read_excel('./data/load/data1001.xlsx')
trainData = df[:df.shape[0]-2364]
#print(trainData)
testData = df[df.shape[0]-2364:df.shape[0]-348]
trainX = trainData.values[:, 1].reshape(500,24)
trainY = trainData.values[:, 1]
trainY = trainY[100:600]
plt.plot(trainData)
plt.show()
#trainData = np.array([trainData_Time, trainData_Temp])

#print(trainY.shape)
#print(trainY)

testX = testData.values[:, 1].reshape(84,24)
testY = testData.values[:, 1]
testY = testY[0:84]
#np.set_printoptions(threshold=np.NaN)


model = create_model(trainX, 5)
model.fit(trainX, trainY, epochs=5000, batch_size=200, verbose=2)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.10f MAE' % (trainScore))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.10f MAE' % (testScore))

# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
plt.plot(trainData)
plt.plot(testY)
#plt.plot(testPredictPlot)
plt.minorticks_on()
plt.show()
