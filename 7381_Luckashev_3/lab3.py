import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import boston_housing


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)

print(test_targets)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


k = 8
num_val_samples = len(train_data) // k
num_epochs = 50
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    H = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(val_data, val_targets))
    all_scores.append(H.history['val_mae'])

    plt.figure(i + 1, (4.6 * 3, 4.8))
    plt.subplot(121)
    plt.plot(range(1, num_epochs + 1), H.history['mae'], label='Training MAE')
    plt.plot(range(1, num_epochs + 1), H.history['val_mae'], label='Validation MAE')
    plt.title('absolute error')
    plt.ylabel('absolute error')
    plt.xlabel('epochs')
    plt.legend()

    plt.subplot(122)
    plt.plot(range(1, num_epochs + 1), H.history['loss'], label='Training loss')
    plt.plot(range(1, num_epochs + 1), H.history['val_loss'], label='Validation loss')
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()

    plt.show()
    plt.clf()


print(np.mean(all_scores[:-1]))

average_mae_history = [np.mean([x[i] for x in all_scores]) for i in range(num_epochs)]
plt.figure(0)
plt.plot(range(1, num_epochs + 1), average_mae_history)
plt.xlabel('epochs')
plt.ylabel("mean absolute error")
plt.title('Mean MAE by blocks')
plt.legend()
plt.show()
