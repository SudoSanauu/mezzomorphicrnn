import numpy as np
import wave 
import math
from byte_converter import *
from sklearn.preprocessing import MinMaxScaler
import os
import glob
#call wave to open audio file

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

#use wave to close the file

# method takes folder of input files or single file as argument
# return an array of all wave files in the folder

def select_files(input_path):
	if os.path.isdir(input_path):
		return glob.glob(input_path +'/*.wav')
	elif os.path.isfile(input_path):
		return [input_path]
	else: 
		raise OSError("file not found")

# method taks list of audio inputs and wave.opens 
# each file transform data to matricies
def files_to_data(file_list):
	data_array = []
	for i in range(len(file_list)): 
		current_file = wave.open(file_list[i], "rb")
		audio_dataset = get_data(current_file)
		scaler = MinMaxScaler(feature_range=(0,1))
		audio_dataset = scaler.fit_transform(audio_dataset)
		data_array.append(audio_dataset)
		current_file.close()
	return data_array


