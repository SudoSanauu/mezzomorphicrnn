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

#n data points at a time
look_back = 512

# open our .wav file and save it as audio_input 
# audio_input = wave.open('chromescale2-24.wav', 'rb')
audio_inputs = select_files(sys.argv[1])

# append all of our bytes to the list audio_dataset
print("Loading all " + str(len(audio_inputs)) +  " file(s)...")
all_datasets = files_to_data(audio_inputs, look_back)

# set our scaling function to normalize dataset
#Take audio_dataset and fit it to our Sigmoid function
#set the size of the dataset we're training on
#set the size of the dataset we're testing on
#create training and testing sections into our dataset
#create dataset method that takes in input dataset and creates a base and expected values


#create training dataset
#create testing dataset
#reshaping train_input
print("reshaping_datasets")
# print(all_datasets[0]['input_dataset'].shape)
# print(all_datasets[1]['input_dataset'].shape)
reshape_datasets(all_datasets)
#reshaping test input

#create an empty model
print("creating_model")
model = Sequential()
#add a LSTM layer to our model
model.add(LSTM(256, input_dim=look_back, dropout_U = .4))
#add output layer
model.add(Dense(1))
#compile model with adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

# print(all_datasets[0]['input_dataset'].shape)
# print(all_datasets[1]['input_dataset'].shape)

#train model on train_input
for d in range(1):
	print("epoch: " + str(d))
	np.random.shuffle(all_datasets)
	for i in range(len(all_datasets)):
		# model.fit(all_datasets[i]['input_dataset'], all_datasets[i]['comparison_dataset'], nb_epoch=1, batch_size=1, verbose=1)
		model.fit(all_datasets[i]['input_dataset'][:100], all_datasets[i]['comparison_dataset'][:100], nb_epoch=1, batch_size=1, verbose=1)


#if user inputs filename, exec, otherwise, default output_file.wav
if len(sys.argv) >= 3:
	audio_output = wave.open(sys.argv[2], 'wb')
else:
	audio_output = wave.open('output_file.wav', 'wb')

# set output length to a reusable variable
output_length = 100
seed_length = look_back + 1 

#ensure output file is same format as a input file
audio_output.setparams(all_datasets[0]['file_params'])
audio_output.setnframes(output_length)


# Choose section for random bunch of notes
# Choose a song
# randomly choose a section from that dataset of length lookback + 5

print("setting seed data")
starting_song = np.random.choice(all_datasets)

random_starting_point = np.random.randint((starting_song['file_params'][3] - seed_length) ,size=1)

seed_data = starting_song['input_dataset'][random_starting_point:random_starting_point + seed_length]
# generation_data = reshape_one_dataset(seed_data)
generation_data = seed_data

print("starting generation")
output_frames = []

# For n times generate a random frame
for i in range(output_length):
	if i % 5000 == 0:
		print("on frame " + str(i) + " of " + str(output_length))

	prediction = model.predict(generation_data, verbose=0)[:look_back]
	# print(generation_data)
	# print(generation_data.shape)
	# print("prediction: ", prediction)
	# print(prediction.shape)
	prediction = reshape_prediction(prediction)
	# print(prediction.shape)
	generation_data = np.append(generation_data, prediction, axis=0)


	predicted_value = prediction[0][0][0]
	# print(predicted_value)
	# print(predicted_value.size)
	# print(output_frames)
	# print(output_frames.size)
	output_frames.append([predicted_value])


	generation_data = generation_data[1:]

print("Finished IO!")

# print(output_frames)

output_frames = starting_song['scaler'].inverse_transform(output_frames)
write_data(audio_output, output_frames)
#close audio files
audio_output.close()
