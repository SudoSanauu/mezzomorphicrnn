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
input_datasets, comparison_datasets = files_to_data(audio_inputs)

# set our scaling function to normalize dataset
#Take audio_dataset and fit it to our Sigmoid function
#set the size of the dataset we're training on
#set the size of the dataset we're testing on
#create training and testing sections into our dataset
#create dataset method that takes in input dataset and creates a base and expected values

#n data points at a time
look_back = 1

#create training dataset
#create testing dataset
#reshaping train_input
input_datasets = reshape_datasets(input_datasets)
#reshaping test input

#create an empty model
model = Sequential()
#add a LSTM layer to our model
model.add(LSTM(16, input_dim=look_back))
#add output layer
model.add(Dense(1))
#compile model with adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

def train_datasets(reshaped_datasets, model):
  np.random.shuffle(reshaped_datasets)
  for i in range(len(reshaped_datasets)):
    model.fit()
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
