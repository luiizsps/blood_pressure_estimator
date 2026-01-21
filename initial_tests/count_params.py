import tensorflow as tf
from tensorflow.keras import layers, models

input_shape = (1250, 1)  # sinal concatenado ECG+PPG

model = models.Sequential([
    layers.Conv1D(128, kernel_size=3, activation='relu', input_shape=input_shape),
    layers.MaxPooling1D(pool_size=10, strides=2),
    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=10, strides=2),
    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=4, strides=2),
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(128)),
    layers.Dense(1)  # saída de regressão
])

model.summary()
print("Total parameters:", model.count_params())
