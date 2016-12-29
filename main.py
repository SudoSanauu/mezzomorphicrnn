import numpy as np
import wave
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys

from byte_converter import *
from io_helper import *

# set random seed
np.random.seed(7)

# set how many data points to accept as input at a time
look_back = 256

# how many times to read over the dataset
epochs = 2

# select which .wav files to open and save them as audio_input
audio_inputs = select_files(sys.argv[1])

# read all data from audio_inputs and save them as an array of
# dictionaries in all_datasets
print("Loading all", str(len(audio_inputs)), "file(s)...")
all_datasets = files_to_data(audio_inputs, look_back)

print("reshaping_datasets")
reshape_datasets(all_datasets)

#create an empty model
print("creating_model")
model = Sequential()
#add a LSTM layer to our model
model.add(LSTM(256, input_dim=look_back, dropout_U = .4))
#add output layer
model.add(Dense(1))
#compile model with adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

#train model on all input datasets randomly shuffled each time
for d in range(epochs):
	print("epoch: " + str(d))
	np.random.shuffle(all_datasets)
	for i in range(len(all_datasets)):
		model.fit(all_datasets[i]['input_dataset'], all_datasets[i]['comparison_dataset'], nb_epoch=1, batch_size=1, verbose=1)


#if user inputs filename open that file, otherwise, default output_file.wav
if len(sys.argv) >= 3:
	audio_output = wave.open(sys.argv[2], 'wb')
else:
	audio_output = wave.open('output_file.wav', 'wb')

# set size of output file and seed for generation
output_length = 300000
seed_length = look_back + 256

#ensure output file is same format as an input file
audio_output.setparams(all_datasets[0]['file_params'])
audio_output.setnframes(output_length)



# choose a song
print("setting seed data")
starting_song = np.random.choice(all_datasets)

# choose starting point
random_starting_point = np.random.randint((starting_song['file_params'][3] - seed_length) ,size=1)

# randomly choose a section from that dataset of length seed_length
seed_data = starting_song['input_dataset'][random_starting_point:random_starting_point + seed_length]
generation_data = seed_data

print("starting generation")
output_frames = []

# For n times generate a random frame
for i in range(output_length):
	if i % 1000 == 0:
		print("on frame " + str(i) + " of " + str(output_length))

	prediction = model.predict(generation_data, verbose=0)[:look_back]
	prediction = reshape_prediction(prediction)
	# Is this what we want? I think we're appending the whole prediction dataset
	# onto our generation data, not just the last one we used. This seems pretty
	# strange to me...
	generation_data = np.append(generation_data, prediction, axis=0)


	predicted_value = prediction[0][0][0]
	output_frames.append([predicted_value])


	generation_data = generation_data[1:]

print("Finished IO!")

output_frames = starting_song['scaler'].inverse_transform(output_frames)
write_data(audio_output, output_frames)

#close audio file
audio_output.close()
