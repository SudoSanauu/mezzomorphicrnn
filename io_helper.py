import numpy as np
import wave 
import math
from byte_converter import *
from sklearn.preprocessing import MinMaxScaler
import os
import glob

#read file and save as one by n matrix
def get_data(wav_file):
	audio_dataset = []
	for i in range(wav_file.getnframes()):
		current_frame = wav_file.readframes(1)
		current_frame = bytes_to_int(current_frame)
		audio_dataset.append([current_frame])
	return audio_dataset

#take in the one by n matrix and save as a file
def write_data(wav_file,prediction_data):
	prediction_int = prediction_data.astype(int)
	width = wav_file.getsampwidth()
	for i in range(len(prediction_int)):
		prediction_bytes = int_to_bytes(prediction_int[i][0], width)
		wav_file.writeframes(prediction_bytes)

# takes folder of input files or single file as argument
# return an array of all wave files in the folder
def select_files(input_path):
	if os.path.isdir(input_path):
		return glob.glob(input_path +'/*.wav')
	elif os.path.isfile(input_path):
		return [input_path]
	else: 
		raise OSError("file not found")

# takes list of audio inputs and wave.opens 
# each file transform data to matrices
def files_to_data(file_list, look_back):
	# create empty output array
	data_dictionary_array = []

	for i in range(len(file_list)): 
		# create new empty dictionary to hold datasets and parameters
		data_dictionary = dict()
		# open new file and store every frame of data
		current_file = wave.open(file_list[i], "rb")
		audio_dataset = get_data(current_file)
		# set the scaler we will use to normalize data for sigmoid function
		scaler = MinMaxScaler(feature_range=(0,1))
		data_dictionary['scaler'] = scaler
		# transform the data with the scaler
		audio_dataset = data_dictionary['scaler'].fit_transform(audio_dataset)
		# create datasets for input and comparison
		input_data, comparison_data = create_dataset(audio_dataset, look_back)
		data_dictionary['input_dataset'] = input_data
		data_dictionary['comparison_dataset'] = comparison_data
		# store parameters of the current file
		data_dictionary['file_params'] = current_file.getparams() 
		# add the dictionary to output array and close the current file
		data_dictionary_array.append(data_dictionary)
		current_file.close()

	return data_dictionary_array

# takes integer audio dataset and converts it into a input
# and comparison np array
def create_dataset(input_dataset, look_back=1):
	input_squence, comparison_sequence = [], []
	for i in range(len(input_dataset)-look_back-1):
		next_input = input_dataset[i:i+look_back, 0]
		input_squence.append(next_input)
		comparison_sequence.append(input_dataset[i+look_back, 0])
	return np.array(input_squence), np.array(comparison_sequence)

# reshape input datasets into a format useable by keras and the  RNN
def reshape_datasets(all_datasets):
	for i in range(len(all_datasets)):
		all_datasets[i]['input_dataset'] = np.reshape(all_datasets[i]['input_dataset'], (all_datasets[i]['input_dataset'].shape[0], 1, all_datasets[i]['input_dataset'].shape[1]))
	return all_datasets

# transform the 
def reshape_prediction(dataset):
	return np.reshape(dataset, (dataset.shape[1], 1, dataset.shape[0]))
