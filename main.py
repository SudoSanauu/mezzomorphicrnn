import numpy as np
import wave
# import scipy
# import matplotlib.pyplot as matplot
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def byte_to_float(input_byte):
	return float(input_byte[0])




# set random seed
np.random.seed(7)

# open our .wav file and save it as audio_io 
audio_io = wave.open('blip.wav', 'rb')

# instantiate an empty list
audio_dataset = list()

# append all of our bytes to the list audio_dataset
for i in range(audio_io.getnframes()):
	current_frame = byte_to_float(audio_io.readframes(1))
	audio_dataset.append([current_frame])

# print(audio_dataset)
# print(type(audio_dataset))

# set our scaling function to normalize dataset
scaler = MinMaxScaler(feature_range=(0,1))
# print(scaler)

audio_dataset = scaler.fit_transform(audio_dataset)

# print(audio_dataset)
# print(type(audio_dataset))


train_size = int(len(audio_dataset) * 0.75)
test_size = len(audio_dataset) - train_size
train, test = audio_dataset[0:train_size,:], audio_dataset[train_size:len(audio_dataset),:]
# print(len(train), len(test))

def create_dataset(input_dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(input_dataset)-look_back-1):
		next_input = input_dataset[i:i+look_back, 0]
		dataX.append(next_input)
		dataY.append(input_dataset[i+look_back, 0])
	return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# print("TrainX: ")
# print(trainX)
# print(type(trainX))
# print(len(trainX))
# print(np.shape(trainX))

# print("TrainY: ")
# print(trainY)
# print(type(trainY))
# print(len(trainY))
# print(np.shape(trainY))

# print("TestX: ")
# print(trainX)
# print(type(trainX))
# print(len(testX))
# print(np.shape(trainX))

# print("TestY: ")
# print(trainY)
# print(type(trainY))
# print(len(testY))
# print(np.shape(trainY))


