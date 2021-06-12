import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential


# odczytanie danych z plików
avocado_train = pd.read_csv('train.csv')
avocado_test = pd.read_csv('test.csv')
avocado_validate = pd.read_csv('dev.csv')


# podzial na X i y
X_train = avocado_train[['average_price', 'total_volume', '4046', '4225', '4770', 'total_bags', 'small_bags', 'large_bags', 'xlarge_bags']]
y_train = avocado_train[['type']]
X_test = avocado_test[['average_price', 'total_volume', '4046', '4225', '4770', 'total_bags', 'small_bags', 'large_bags', 'xlarge_bags']]
y_test = avocado_test[['type']]

print(X_train.shape[1])
# keras model
model = Sequential()
model.add(Dense(9, input_dim = X_train.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(1,kernel_initializer='normal', activation='sigmoid'))

early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)

# kompilacja
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model fit
# epochs = int(sys.argv[1])
# batch_size = int(sys.argv[2])
epochs = 10
batch_size = 128

# trenowanie modelu
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))


# prediction
predictions = model.predict(X_test)
#pd.DataFrame(predictions).to_csv('prediction_results.csv')

# ewaluacja
error = mean_squared_error(y_test, predictions)
print('Error: ', error)
