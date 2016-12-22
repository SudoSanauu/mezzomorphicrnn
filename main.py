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
all_datasets = files_to_data(audio_inputs)

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
reshape_datasets(all_datasets)
#reshaping test input

#create an empty model
model = Sequential()
#add a LSTM layer to our model
model.add(LSTM(16, input_dim=look_back))
#add output layer
model.add(Dense(1))
#compile model with adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

#train model on train_input
for d in range(5):
  np.random.shuffle(all_datasets)
  for i in range(len(all_datasets)):
    model.fit(all_datasets[i]['input_dataset'], all_datasets[i]['comparison_dataset'], nb_epoch=1, batch_size=1, verbose=1)


#use model to predict for training and testing data
prediction = model.predict(all_datasets[0]['input_dataset'])
comparison = all_datasets[0]['comparison_dataset']
# unscale our data so that we can transform them back into bytes
prediction = all_datasets[0]['scaler'].inverse_transform(prediction)
comparison = all_datasets[0]['scaler'].inverse_transform(comparison)

#if user inputs filename, exec, otherwise, default output_file.wav
if len(sys.argv) >= 3:
	audio_output = wave.open(sys.argv[2], 'wb')
else:
	audio_output = wave.open('output_file.wav', 'wb')

#ensure output file is same format as input file
audio_output.setparams(all_datasets[0]['file_params'])
# audio_output.setnframes(len(train_predict) + len(test_predict))

#write train and test to file
write_data(audio_output, prediction)

#close audio files
audio_output.close()
