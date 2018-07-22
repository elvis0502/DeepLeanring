import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM

np.random.seed(0)

def create_model(inputs, neurons, dropout =0.25):
    model = Sequential()
    model.add(Dense(100, input_dim=24, activation='relu'))
    #model.add(LSTM(neurons, input_shape=(30559, 2)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    #model.add(Activation(activ_func))
    model.compile(loss="mae", optimizer="adam")
    return model

df = pd.read_csv('./data/Small.csv')
#print(df.shape[0])
trainData = df[0:df.shape[0]-20000]
testData = df[df.shape[0]-1000:]


predictors = ['Usage', 'Temp', 'Hour']
Xtrain = trainData[predictors]
Xtrain = Xtrain.values
Ytrain = trainData['Usage']
Ytrain = Ytrain.values
#print(Ytrain)
#Xtrain = Xtrain.reshape(30559, 1, 2)
#print(Xtrain)
testPredictors = ['Temp', 'Hour']
Xtest = testData[predictors]
Xtest = Xtest.values
Ytest = testData['Usage']
Ytest = Ytest.values

model = create_model(Xtrain, 5)
model.fit(Xtrain, Ytrain, epochs=50, batch_size=200, verbose=2)

# Estimate model performance
trainScore = model.evaluate(Xtrain, Ytrain, verbose=0)
print('Train Score: %.10f MAE' % (trainScore))
testScore = model.evaluate(Xtrain, Ytrain, verbose=0)
print('Test Score: %.10f MAE' % (testScore))

# generate predictions for training
trainPredict = model.predict(Xtrain)
testPredict = model.predict(Xtest)
plt.plot(testPredict)
plt.plot(Ytest)
#plt.plot(testPredictPlot)
plt.minorticks_on()
plt.show()

'''
# shift train predictions for plotting
combined_data = np.append(trainData, testData)
trainPredictPlot = np.empty((len(combined_data), 1))
trainPredictPlot[:] = np.nan
#trainPredictPlot[look_back:len(trainPredict) + look_back] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty((len(combined_data), 1))
testPredictPlot[:] = np.nan
#testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(combined_data) - 1] = testPredict

# Combine the results
#combined_df = train_df.append(test_df)
#combined_dates = combined_df['DATE']

# plot baseline and predictions
plt.plot(combined_data)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.minorticks_on()
plt.show()
'''

'''
fig = plt.figure(1, figsize=[15,10])
df['Temp'].plot()
plt.ylabel('1')
plt.xlabel('2')
plt.title('Electricity consumption forecast')
plt.legend()
plt.show()

test['Usage'].plot()
plt.ylabel('1')
plt.xlabel('2')
plt.title('Electricity consumption forecast')
plt.legend()
plt.show()
'''
'''
print(trainData)
print(testData)
predictors = ['Usage', 'Temp', 'Hour']
Xtrain = trainData[predictors]
Ytrain = trainData['Usage']
Xtest = testData[predictors]
Ytest = testData['Usage']

def build5():
    model = Sequential(name='power')
    model.add(Dense(100, input_dim=3, activation='relu'))

    model.add(Dense(1))
    return model

for i in range(5,6):
    model=eval("build"+str(i)+"()")
    model.compile('adam','mae')#3.0895
    history=model.fit(Xtrain,Ytrain,batch_size=404,epochs=500,verbose=1,validation_data=(Xtest,Ytest))
    print(history.history)

    f=open("result.txt",'a')
    f.write(str(history.history['val_loss'][-1])+"\n")
    f.close()
    '''