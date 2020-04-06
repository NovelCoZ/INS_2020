import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Input, Dense

n_row = 600

# generate n_row rows of data
data = np.zeros((n_row, 6))
targets = np.zeros(n_row)

for i in range(n_row):
    x = np.random.normal(-5, 10)
    e = np.random.normal(0, 0.3)
    data[i, :] = (
        - x ** 3 + e,
        np.log(np.fabs(x)) + e,
        np.sin(3 * x) + e,
        np.exp(x) + e,
        -x + np.sqrt(np.fabs(x)) + e,
        x + e
    )
    targets[i] = x + 4 + e

# normalize data
data -= data.mean(axis=0)
data /= data.std(axis=0)
targets -= targets.mean(axis=0)
targets /= targets.std(axis=0)

# create functional regression model with auto encoder.
# input -> encoder -> (decoder, regression)
main_input = Input(shape=(6,))

encoder = Dense(64, activation='relu')(main_input)
encoder = Dense(32, activation='relu')(encoder)
encoder = Dense(3, activation='relu')(encoder)

decoder = Dense(32, activation='relu', kernel_initializer='normal', name='decoder_hidden1')(encoder)
decoder = Dense(64, activation='relu', name='decoder_hidden2')(decoder)
decoder = Dense(6, name='decoder')(decoder)

predictor = Dense(64, kernel_initializer='normal', activation='relu')(encoder)
predictor = Dense(64, activation='relu')(predictor)
predictor = Dense(64, activation='relu')(predictor)
predictor = Dense(1, name="prediction")(predictor)

model = Model(main_input, [decoder, predictor])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# train model
model.fit(data, [data, targets], epochs=250, validation_split=0.2)

# divide model on three: encoder, decoder, regression
encoder_model = Model(main_input, encoder)
regression_model = Model(main_input, predictor)

# Model input can only be instance of Input layer.
# Cannot use encoder layer, so here is little trick,
# where we use already existing layers to create new functional model
encoded_input = Input(shape=(3,))
decoder2 = model.get_layer('decoder_hidden1')(encoded_input)
decoder2 = model.get_layer('decoder_hidden2')(decoder2)
decoder2 = model.get_layer('decoder')(decoder2)
decoder_model = Model(encoded_input, decoder2)

# save models
encoder_model.save('encoder.h5')
decoder_model.save('decoder.h5')
regression_model.save('regression.h5')

# get & save necessary data
encoded_data = encoder_model.predict(data)
decoded_data = decoder_model.predict(encoded_data)
regression = regression_model.predict(data)

pd.DataFrame(data).to_csv("data.csv")
pd.DataFrame(encoded_data).to_csv("encoded_data.csv")
pd.DataFrame(decoded_data).to_csv("decoded_data.csv")
pd.DataFrame(targets).to_csv("actual_targets.csv")
pd.DataFrame(regression).to_csv("predicted_targets.csv")
