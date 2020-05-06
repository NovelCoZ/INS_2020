from var3 import gen_data
from datetime import datetime
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random


class SaveModel(Callback):
    def __init__(self, epochs_to_save, user_prefix):
        self.prefix = str(datetime.now().strftime("%d-%m-%Y")) + "_" + user_prefix + "_"
        self.epochs_to_save = epochs_to_save

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epochs_to_save:
            self.model.save(self.prefix + str(epoch))


samples, size = 500, 50

data, labels = gen_data(samples, size)
rand = list(range(len(data)))
random.shuffle(rand)
data = data[rand]
labels = labels[rand]

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
labels = to_categorical(labels, 2)

data = data.reshape(data.shape[0], size, size, 1)

epochs_to_save = [1, 2, 4, 6, 11, 12]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(size, size, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
H = model.fit(data, labels,
                    batch_size=20,
                    epochs=12,
                    validation_split=0.2,
              callbacks=[SaveModel(epochs_to_save, 'epoch')])

loss = H.history['loss']
val_loss = H.history['val_loss']
acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
epochs = range(1, len(loss) + 1)

# Построение графика ошибки
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Построение графика точности
plt.clf()
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
