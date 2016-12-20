import numpy as np
import wave 
import math
from byte_converter import *

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
