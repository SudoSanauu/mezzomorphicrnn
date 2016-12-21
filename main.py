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
audio_inputs = select_files(sys.argv[1])

# append all of our bytes to the list audio_dataset
audio_datasets = files_to_data(audio_inputs)

# set our scaling function to normalize dataset
#Take audio_dataset and fit it to our Sigmoid function

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

#reshaping train_input
train_input = np.reshape(train_input, (train_input.shape[0], 1, train_input.shape[1]))

#reshaping test input
test_input = np.reshape(test_input, (test_input.shape[0], 1, test_input.shape[1]))

#create an empty model
model = Sequential()

#add a LSTM layer to our model
model.add(LSTM(16, input_dim=look_back))

#add output layer
model.add(Dense(1))

#compile model with adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

#train model on train_input
model.fit(train_input, train_comparison, nb_epoch=1, batch_size=1, verbose=1)

#use model to predict for training and testing data
train_predict = model.predict(train_input)
test_predict = model.predict(test_input)

# unscale our data so that we can transform them back into bytes
train_predict = scaler.inverse_transform(train_predict)
train_comparison = scaler.inverse_transform(train_comparison)
test_predict = scaler.inverse_transform(test_predict)
test_comparison = scaler.inverse_transform(test_comparison)

#if user inputs filename, exec, otherwise, default output_file.wav
if len(sys.argv) >= 3:
	audio_output = wave.open(sys.argv[2], 'wb')
else:
	audio_output = wave.open('output_file.wav', 'wb')

#ensure output file is same format as input file
audio_output.setparams(audio_input.getparams())
# audio_output.setnframes(len(train_predict) + len(test_predict))

#write train and test to file
write_data(audio_output,train_predict)
write_data(audio_output,test_predict)

#close audio files
audio_output.close()
audio_input.close()
