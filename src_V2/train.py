# import numpy as np
# import pickle
from data import load_data
from model import Model, Convolution, MaxPooling, Affine
import pickle

X_train, X_test, y_train, y_test = load_data("./dataset")
batch_size = 32
model = Model()
model.add_layers(Convolution(filter_shape=(3, 1, 3, 3), padding=1, stride=1, activation='relu'),
                 MaxPooling((2, 2), stride=2),
                 Convolution(filter_shape=(1, 3, 3, 3), padding=0, stride=1, activation="relu"),
                 MaxPooling((2, 2), stride=2),
                 Affine())
model.set_parameters()
model.fit(X_train, y_train, X_test, y_test, epochs=500, batch_size=50, learning_rate=0.001)

with open('../ckpt/ckpt.pkl', 'wb') as f:
    pickle.dump(model, f)
