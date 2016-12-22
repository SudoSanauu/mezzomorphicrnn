import numpy as np
import wave
import scipy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import sys
from random import randint


from byte_converter import *
from io_helper import *

# set random seed
np.random.seed(7)

# open our .wav file and save it as audio_input
# audio_input = wave.open('chromescale2-24.wav', 'rb')
audio_input = wave.open(sys.argv[1], 'rb')

# instantiate an empty list
audio_dataset = list()
unique_bytes = list()

# append all of our bytes to the list audio_dataset
for i in range(audio_input.getnframes()):
	current_frame = audio_input.readframes(1)
	unique_bytes.append(current_frame)
	current_frame = bytes_to_int(current_frame)
	audio_dataset.append([current_frame])

# build a unique byte set and initiate conversion dicts
byte_set = sorted(list(set(unique_bytes)))
byte_to_int = dict()
int_to_byte = dict()

# build conversion dicts
count = 0
for i in byte_set:
	byte_to_int[byte_set[count]] = bytes_to_int(byte_set[count])
	int_to_byte[bytes_to_int(byte_set[count])] = byte_set[count]
	count+=1

# find our song length and number of "notes"
n_bytes = len(audio_dataset)
n_notes = len(byte_set)
print ("Total Bytes: ", n_bytes)
print ("Total 'Notes': ", n_notes)

# append all of our bytes to the list audio_dataset
# audio_dataset = get_data(audio_input)

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
look_back = 500

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
model.add(LSTM(256, input_dim=look_back, dropout_U = .4 ,dropout_W = .4))

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

# pick random starting "note"
random_dataset = list()
for i in range(look_back+5):
	note = byte_to_int[byte_set[randint(0,len(byte_set)-1)]]
	random_dataset.append([note])

generator = scaler.fit_transform(random_dataset)
width = audio_output.getsampwidth()

# range is equal to length of song output
for i in range(1000):
	# Build Dataset
	generator = scaler.fit_transform(generator)
	generatedX, generatedY = create_dataset(generator, look_back)
	# Scale Dataset
	# Reshape Dataset for consumption
	generatedX = np.reshape(generatedX, (generatedX.shape[0], 1, generatedX.shape[1]))
	# predict next value
	prediction = model.predict(generatedX, verbose=0)
	#grab the predicted value
	predicted = prediction[0][0]
	#append the predicted value
	generator = np.append(generator, [[predicted]], axis=0)
	#get the result
	result = predicted*(256**audio_input.getsampwidth())
	#write the next frame based on the result
	audio_output.writeframes(int_to_bytes(int(result), audio_input.getsampwidth()))
	#generator for input now equals everthing from 1 after the start value to the new value
	generator = generator[1:len(generator)]
print ("\nDone.")

#write train and test to file
# write_data(audio_output,train_predict)
# write_data(audio_output,test_predict)

#close audio files
audio_output.close()
audio_input.close()
