import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')

data = [df['x'].to_list(), df['y'].to_list()]
training_threshold = int(0.7*len(data[0]))
x = [v for v in data[0]]
y = [v for v in data[1]]
print(x)
print(y)

training_x = x[:training_threshold]
training_y = y[:training_threshold]

test_x = x[training_threshold:]
test_y = y[training_threshold:]

model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(1,), activation='relu'),
    keras.layers.Dense(1)
])
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')
model.fit(training_x, training_y, epochs=1000)

print(model.predict(test_x))
#model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error', metrics=['mse'])


