import unittest
import wave
import os
import numpy as np
from io_helper import *

class IOHelperTestCase(unittest.TestCase):

	# testing get_data
	# checking to see if we get something back
	def test_get_audio(self):
		input_file = wave.open('blip.wav','rb')
		audio_input = get_data(input_file)
		x,y = np.shape(audio_input)
		self.assertEqual(y,1)
		#take the type of something and compare
		self.assertEqual(type(audio_input[0][0]),type(1))
		input_file.close()

	# testing write_data
	# put in simple data for it to write
	def test_write_data(self):
		input_file = wave.open('blip.wav','rb')
		file_params = input_file.getparams()
		output_file = wave.open('test_output_file.wav','wb')
		output_file.setparams(input_file.getparams())
		input_file.close()
		#write to the file
		mock_data = np.array([[1] , [2]]) 
		write_data(output_file,mock_data)
		output_file.close()
		os.remove('test_output_file.wav')

	# read and write a file and compare
	def test_read_to_write(self):
		input_file = wave.open('blip.wav','rb')
		file_params = input_file.getparams()
		output_file = wave.open('test_output_file.wav','wb')
		output_file.setparams(input_file.getparams())
		audio_input = get_data(input_file)
		audio_input = np.array(audio_input)
		write_data(output_file,audio_input)
		input_file.close()
		output_file.close()
		output_file = wave.open('test_output_file.wav','rb')
		audio_output = get_data(output_file)
		audio_output = np.array(audio_input)
		self.assertEqual(np.array_equal(audio_input,audio_output), True)

	# testing select_files



	# testing files_to_data


	# testing create_dataset


	# testing reshape_dataset


	# testing reshape_predict
