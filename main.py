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

# set random seed
np.random.seed(7)

# open our .wav file and save it as audio_input 
# audio_input = wave.open('chromescale2-24.wav', 'rb')
audio_input = wave.open(sys.argv[1], 'rb')

# append all of our bytes to the list audio_dataset
audio_dataset = get_data(audio_input)

# set our scaling function to normalize dataset
scaler = MinMaxScaler(feature_range=(0,1))

#Take audio_dataset and fit it to our Sigmoid function
audio_dataset = scaler.fit_transform(audio_dataset)

#set the size of the dataset we're training on
train_size = int(len(audio_dataset) * 0.5)

#set the size of the dataset we're testing on
test_size = len(audio_dataset) - train_size

#create training and testing sections into our dataset
train, test = audio_dataset[0:train_size,:], audio_dataset[train_size:len(audio_dataset),:]

#create dataset method that takes in input dataset and creates a base and expected values
def create_dataset(input_dataset, look_back=1):
	input_squence, comparison_sequence = [], []
	for i in range(len(input_dataset)-look_back-1):
		next_input = input_dataset[i:i+look_back, 0]
		input_squence.append(next_input)
		comparison_sequence.append(input_dataset[i+look_back, 0])
	return np.array(input_squence), np.array(comparison_sequence)

#n data points at a time
look_back = 1

#create training dataset
train_input, train_comparison = create_dataset(train, look_back)

#create testing dataset
test_input, test_comparison = create_dataset(test, look_back)

train_input = np.reshape(train_input, (train_input.shape[0], 1, train_input.shape[1]))
test_input = np.reshape(test_input, (test_input.shape[0], 1, test_input.shape[1]))

# print("TrainX: ")
# print(train_input)
# print("TestX: ")
# print(test_input)



model = Sequential()

model.add(LSTM(16, input_dim=look_back))

model.add(Dense(1))

# CHECK WHAT OPTIMIZER MEANS
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

model.fit(train_input, train_comparison, nb_epoch=1, batch_size=1, verbose=1)

train_predict = model.predict(train_input)
test_predict = model.predict(test_input)

# print("train_predict")
# print(train_predict)
# print("test_predict")
# print(test_predict)

# unscale our data so that we can transform them back into bytes
train_predict = scaler.inverse_transform(train_predict)
train_comparison = scaler.inverse_transform(train_comparison)
test_predict = scaler.inverse_transform(test_predict)
test_comparison = scaler.inverse_transform(test_comparison)

# print("train_predict")
# print(train_predict)
# print("test_predict")
# print(test_predict)

# print("train_comparison[0]: ")
# print(train_comparison[0])

# train_score = math.sqrt(mean_squared_error(train_comparison[0], train_predict[:,0]))
# print('Train Score: %.2f RMSE' % (train_score))
# test_score = math.sqrt(mean_squared_error(test_comparison[0], test_predict[:,0]))
# print('Train Score: %.2f RMSE' % (test_score))

if len(argv) >= 3:
	audio_output = wave.open(argv[2], 'wb')
else:
	audio_output = wave.open('output_file.wav', 'wb')


audio_output.setparams(audio_input.getparams())
# audio_output.setnframes(len(train_predict) + len(test_predict))

write_data(audio_output,train_predict)
write_data(audio_output,test_predict)

audio_output.close()
audio_input.close()
