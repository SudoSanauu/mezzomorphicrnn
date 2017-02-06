import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler


random_dataset = list()
for i in range(200):
	random_dataset.append([randint(0,(256**1))])

scaler = MinMaxScaler(feature_range=(0,1))
random_dataset = scaler.fit_transform(random_dataset)
pattern = random_dataset
for_append = [5]

print (len(pattern))
pattern = np.append(pattern, for_append[0:1])
print (pattern[-1])
