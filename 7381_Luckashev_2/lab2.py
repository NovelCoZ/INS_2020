import numpy as np
import pandas
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
np.random.shuffle(dataset)

X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print(encoded_Y)

for i in range(3):
    _X = X
    model = Sequential()
    if i == 0:
        model.add(Dense(60, init='normal', activation='relu'))
    if i == 1:
        _X = _X[:, 0:30]
        model.add(Dense(30, init='normal', activation='relu'))
    if i == 2:
        model.add(Dense(60, init='normal', activation='relu'))
        model.add(Dense(15, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    H = model.fit(_X, encoded_Y, epochs=100, batch_size=10, validation_split=0.1, verbose=False)
    print(model.summary())
    loss = H.history['loss']
    val_loss = H.history['val_loss']
    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    plt.figure(None, (4.6 * 3, 4.8))
    # Построение графика ошибки
    plt.subplot(121)
    plt.plot(epochs, loss, 'r.', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Построение графика точности
    plt.subplot(122)

    plt.plot(epochs, acc, 'r.', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.clf()
