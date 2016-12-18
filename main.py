import numpy as np
import wave
import scipy
import matplotlib.pyplot as plot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math


def byte_to_float(input_byte):
	return float(input_byte[0])




# set random seed
np.random.seed(7)

# open our .wav file and save it as audio_io 
audio_io = wave.open('blip.wav', 'rb')

# instantiate an empty list
audio_dataset = list()

# append all of our bytes to the list audio_dataset
for i in range(audio_io.getnframes()):
	current_frame = byte_to_float(audio_io.readframes(1))
	audio_dataset.append([current_frame])

# print(audio_dataset)
# print(type(audio_dataset))

# set our scaling function to normalize dataset
scaler = MinMaxScaler(feature_range=(0,1))
# print(scaler)

audio_dataset = scaler.fit_transform(audio_dataset)

# print(audio_dataset)
# print(type(audio_dataset))


train_size = int(len(audio_dataset) * 0.75)
test_size = len(audio_dataset) - train_size
train, test = audio_dataset[0:train_size,:], audio_dataset[train_size:len(audio_dataset),:]
# print(len(train), len(test))

def create_dataset(input_dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(input_dataset)-look_back-1):
		next_input = input_dataset[i:i+look_back, 0]
		dataX.append(next_input)
		dataY.append(input_dataset[i+look_back, 0])
	return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)


# print("TrainX: ")
# print(trainX)
# print(type(trainX))
# print(len(trainX))
# print(np.shape(trainX))

# print("TrainY: ")
# print(trainY)
# print(type(trainY))
# print(len(trainY))
# print(np.shape(trainY))

testX, testY = create_dataset(test, look_back)

# print("TestX: ")
# print(trainX)
# print(type(trainX))
# print(len(testX))
# print(np.shape(trainX))

# print("TestY: ")
# print(trainY)
# print(type(trainY))
# print(len(testY))
# print(np.shape(trainY))


trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# print("TrainX: ")
# print(trainX)
# print("TestX: ")
# print(testX)



model = Sequential()

model.add(LSTM(32, input_dim=look_back))

model.add(Dense(1))

# CHECK WHAT OPTIMIZER MEANS
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

model.fit(trainX, trainY, nb_epoch=1, batch_size=1, verbose=1)

train_predict = model.predict(trainX)
test_predict = model.predict(testX)

# print("train_predict")
# print(train_predict)
# print("test_predict")
# print(test_predict)

train_predict = scaler.inverse_transform(train_predict)
trainY = scaler.inverse_transform(trainY)
test_predict = scaler.inverse_transform(test_predict)
testY = scaler.inverse_transform(testY)

# print("train_predict")
# print(train_predict)
# print("test_predict")
# print(test_predict)

# print("trainY[0]: ")
# print(trainY[0])

# train_score = math.sqrt(mean_squared_error(trainY[0], train_predict[:,0]))
# print('Train Score: %.2f RMSE' % (train_score))
# test_score = math.sqrt(mean_squared_error(testY[0], test_predict[:,0]))
# print('Train Score: %.2f RMSE' % (test_score))









# shift train predictions for plotting
train_predict_plot = np.empty_like(audio_dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
test_predict_plot = np.empty_like(audio_dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(look_back*2)+1:len(audio_dataset)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(audio_dataset))
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.show()