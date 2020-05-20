import numpy as np
import re
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def build_models():
    models = []
    model1 = Sequential()
    model1.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model1.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model1.add(MaxPooling1D(pool_size=2))
    model1.add(Dropout(0.3))
    model1.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model1.add(MaxPooling1D(pool_size=2))
    model1.add(Dropout(0.3))
    model1.add(LSTM(100))
    model1.add(Dropout(0.3))
    model1.add(Dense(1, activation='sigmoid'))

    model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model2 = Sequential()
    model2.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model2.add(LSTM(100))
    model2.add(Dropout(0.3))
    model2.add(Dense(100, activation='relu'))
    model2.add(Dropout(0.4))
    model2.add(Dense(1, activation='sigmoid'))

    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    models.append(model1)
    models.append(model2)

    return models


def train_models(train_x, train_y, test_x, test_y):
    models = build_models()
    model1 = models[0]
    model2 = models[1]

    plot_history(model1.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs1, batch_size=batch_size1))
    print(model1.evaluate(test_x, test_y, verbose=0))
    model1.save('model1.h5')

    plot_history(model2.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs2, batch_size=batch_size2))
    print(model2.evaluate(test_x, test_y, verbose=0))
    model2.save('model2.h5')
    return [model1, model2]


def test_ensemble():
    a = model1.predict(test_x)
    b = model2.predict(test_x)
    prediction = np.divide(np.add(a, b), 2)
    predictions = np.greater_equal(prediction, np.array([0.5]))
    targets = np.reshape(test_y, (np.size(predictions), 1))
    accuracy = np.mean(np.logical_not(np.logical_xor(predictions, targets)))
    ensemble_info = "Ensemble accuracy: " + str(accuracy)
    print(ensemble_info)


def ensemble(to_predict=None):
    if to_predict.all() is not None:
        return "Ensemble predict: " + str(np.mean((model1.predict(to_predict)[0][0], model2.predict(to_predict)[0][0])))


def ensemble_predict(text):
    output_str = ''
    with open(text, 'r') as f:
        for input_str in f.readlines():
            output_str += re.sub('[^A-Za-z0-9 ]+', '', input_str).lower()
    indexes = imdb.get_word_index()
    encode = []
    text = output_str.split()
    for index in text:
        if index in indexes and indexes[index] < 10000:
            encode.append(indexes[index])
    encode = sequence.pad_sequences([np.array(encode)], maxlen=max_review_length)
    print(ensemble(encode))


embedding_vector_length = 32
top_words = 10000
max_review_length = 500
batch_size1 = 100
batch_size2 = 64
epochs1 = 3
epochs2 = 3

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=top_words)

training_data = sequence.pad_sequences(training_data, maxlen=max_review_length)
testing_data = sequence.pad_sequences(testing_data, maxlen=max_review_length)

data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

[model1, model2] = train_models(train_x, train_y, test_x, test_y)
#model1 = load_model("model1.h5")
#model2 = load_model("model2.h5")
test_ensemble()
ensemble_predict('test.txt')
