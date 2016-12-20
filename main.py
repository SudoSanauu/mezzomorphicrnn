import numpy as np
import wave
import scipy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import sys

from byte_converter import *
from io_helper import *

# import pdb




# set random seed
np.random.seed(7)

# open our .wav file and save it as audio_input 
# audio_input = wave.open('chromescale2-24.wav', 'rb')
audio_input = wave.open(sys.argv[1], 'rb')

# append all of our bytes to the list audio_dataset
audio_dataset = get_data(audio_input)

# print(audio_dataset)
# print(type(audio_dataset))

# set our scaling function to normalize dataset
scaler = MinMaxScaler(feature_range=(0,1))
# print(scaler)

audio_dataset = scaler.fit_transform(audio_dataset)

# print(audio_dataset)
# print(type(audio_dataset))

# pdb.set_trace()



train_size = int(len(audio_dataset) * 0.5)
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

model.add(LSTM(16, input_dim=look_back))

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

# unscale our data so that we can transform them back into bytes
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









# # shift train predictions for plotting
# train_predict_plot = np.empty_like(audio_dataset)
# train_predict_plot[:, :] = np.nan
# train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
# # shift test predictions for plotting
# test_predict_plot = np.empty_like(audio_dataset)
# test_predict_plot[:, :] = np.nan
# test_predict_plot[len(train_predict)+(look_back*2)+1:len(audio_dataset)-1, :] = test_predict
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(audio_dataset))
# plt.plot(train_predict_plot)
# plt.plot(test_predict_plot)
# plt.show()



audio_output = wave.open('debugged_output.wav', 'wb')

audio_output.setparams(audio_input.getparams())
# audio_output.setnframes(len(train_predict) + len(test_predict))

write_data(audio_output,train_predict)
write_data(audio_output,test_predict)


# train_predict_int = train_predict.astype(int)
# test_predict_int = test_predict.astype(int)
# width = audio_output.getsampwidth()

# print(train_predict_int)
# print(test_predict_int)



# for i in range(len(train_predict)):
# 	train_predict_bytes = int_to_bytes(train_predict_int[i][0],width)
# 	# print((train_predict_bytes))
# 	audio_output.writeframes(train_predict_bytes)

# for i in range(len(test_predict)):
# 	test_predict_bytes = int_to_bytes(test_predict_int[i][0], width)
# 	# print((train_predict_bytes))
# 	audio_output.writeframes(test_predict_bytes)


audio_output.close()
audio_input.close()
