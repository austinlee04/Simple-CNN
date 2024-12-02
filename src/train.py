# import numpy as np
# import pickle
from data import load_data
from model import SimpleCNN, Convolution, MaxPooling, FullyConnectedLayer

X_train, X_test, y_train, y_test = load_data("./dataset")
batch_size = 32
CNN_model = SimpleCNN()
CNN_model.add_layers(Convolution(filter_shape=(1, 1, 5, 5), stride=2, activation="relu"),
                     MaxPooling(PH=4, PW=4, stride=4),
                     FullyConnectedLayer(activation='softmax'))
CNN_model.set_parameters()
CNN_model.fit(X_train, y_train, epochs=100, batch_size=100, learning_rate=0.01)
# print(CNN_model.score(X_train[:20], y_train[:20]))
'''
CNN_model.fit(X_train, y_train, batch_size=batch_size)
print(f"train score : {CNN_model.score(X_train, y_train)}\ntest score : {CNN_model.score(X_test, y_test)}")

CNN_model.save_model()
'''
