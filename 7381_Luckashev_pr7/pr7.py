import datetime
from keras.models import Sequential
from keras import layers, callbacks
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def gen_sequence(seq_len=1000):
    seq = [math.sin(i / 5) * math.sin(i / 10 + 0.5) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)


def draw_sequence():
    seq = gen_sequence(100)
    plt.plot(range(len(seq)), seq)
    plt.show()


draw_sequence()


def gen_data_from_sequence(seq_len=1006, lookback=10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i, i+lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return past, future


data, res = gen_data_from_sequence()

dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size+val_size], res[train_size:train_size+val_size]
test_data, test_res = data[train_size+val_size:], res[train_size+val_size:]

model = Sequential()
model.add(layers.GRU(16, recurrent_activation='sigmoid', input_shape=(None, 1),
                     return_sequences=True))
model.add(layers.LSTM(16, activation='sigmoid', input_shape=(None, 1),
                      return_sequences=True, dropout=0.5))
model.add(layers.GRU(8, input_shape=(None, 1)))
model.add(layers.Dense(1))

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer='nadam', loss='mse')
history = model.fit(train_data, train_res, epochs=40,
                    validation_data=(val_data, val_res), callbacks=[tensorboard_callback])

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(len(loss)), loss, label="Training")
plt.plot(range(len(val_loss)), val_loss, label="Validation")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

predicted_res = model.predict(test_data)
pred_length = range(len(predicted_res))
plt.title('Predicted vs actual values')
plt.plot(pred_length, test_res, label="Actual")
plt.plot(pred_length, predicted_res, label="Predicted")
plt.legend()
plt.show()
