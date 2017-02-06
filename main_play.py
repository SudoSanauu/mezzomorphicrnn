import numpy as np
import wave
import scipy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import sys
from random import randint


from byte_converter import *

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

# set our scaling function to normalize dataset
scaler = MinMaxScaler(feature_range=(0,1))
audio_dataset = scaler.fit_transform(audio_dataset)

# set training and test sizes, and build arrays
train_size = int(len(audio_dataset) * 0.5)
test_size = len(audio_dataset) - train_size
train, test = audio_dataset[0:train_size,:], audio_dataset[train_size:len(audio_dataset),:]

# create datasets
def create_dataset(input_dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(input_dataset)-look_back-1):
		next_input = input_dataset[i:i+look_back, 0]
		dataX.append(next_input)
		dataY.append(input_dataset[i+look_back, 0])
	return np.array(dataX), np.array(dataY)

# define look back and call dataset method for training
look_back = 1000
trainX, trainY = create_dataset(train, look_back)

# create dataset for testing
testX, testY = create_dataset(test, look_back)

# reshape datasets for NN consumption
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# Declare our NN will be an RNN
model = Sequential()
# Build LSTM layer
model.add(LSTM(256, input_dim=look_back, dropout_U = .45, dropout_W = .45))
# model.add(Dropout(0.2))
model.add(Dense(1))

# CHECK WHAT OPTIMIZER MEANS
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

# Save weights from training session
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Train our model
model.fit(trainX, trainY, nb_epoch=1, batch_size=1, verbose=1)

# Run predictions
# train_predict = model.predict(trainX)
# test_predict = model.predict(testX)
#
# # unscale our data so that we can transform them back into bytes
# train_predict = scaler.inverse_transform(train_predict)
# trainY = scaler.inverse_transform(trainY)
# test_predict = scaler.inverse_transform(test_predict)
# testY = scaler.inverse_transform(testY)

# pick random starting "note"
random_dataset = list()
for i in range(1100):
	note = byte_to_int[byte_set[randint(0,len(byte_set)-1)]]
	# print(note)
	random_dataset.append([note])

random_dataset = scaler.fit_transform(random_dataset)
generator = random_dataset

# open wave files and set parameters
audio_output = wave.open('friedricelongdropout.wav', 'wb')
audio_output.setparams(audio_input.getparams())
width = audio_output.getsampwidth()

# range is equal to length of song output
for i in range(300000):
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




#
# for i in range(len(train_predict)):
# 	train_predict_bytes = int_to_bytes(train_predict_int[i][0],width)
# 	# print((train_predict_bytes))
# 	audio_output.writeframes(train_predict_bytes)
#
# for i in range(len(test_predict)):
# 	test_predict_bytes = int_to_bytes(test_predict_int[i][0], width)
# 	# print((train_predict_bytes))
# 	audio_output.writeframes(test_predict_bytes)

# close files
audio_output.close()
audio_input.close()
