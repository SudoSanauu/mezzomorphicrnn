from sklearn.preprocessing import MinMaxScaler


data = [245, 123, 593]
scaler = MinMaxScaler(feature_range=(0,1))
data_transformed = scaler.fit_transform(data)
print(data_transformed)
