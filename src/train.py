import numpy as np
import pickle
from data import load_data
from model import SimpleCNN, Convolution, Pooling

X_train, X_test, y_train, y_test = load_data("./dataset")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(y_train[0])

CNN_model = SimpleCNN(learning_rate=0.01)
CNN_model.add_layer()
CNN_model.fit(X_train, y_train)
print(f"train score : {CNN_model.score(X_train, y_train)}\ntest score : {CNN_model.score(X_test, y_test)}")

CNN_model.save_model()
