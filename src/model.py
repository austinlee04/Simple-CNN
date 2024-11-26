import numpy as np
import pickle

class SimpleCNN:
    def __init__(self, learning_rate, batch_size):
        # general parameters for learning
        self.lr = learning_rate
        self.batch_size = batch_size

        # parameters for convolutional layer
        self.input_size = (self.batch_size, 1, 28, 28)
        self.filter_num = 3
        self.filter_size = (self.filter_num, 1, 3, 3)
        self.bias_size = (self.filter_num, 1, 1)
        self.output_size = (1, 10, 1)

        self.filter = 0
        self.padding = 0
        self.stride = 1

        # parameters for linear layer
        self.w = 0
        self.b = 0

    def im2col(self, X):
        pass

    def fit(self, X_train, y_train):
        for i in range(0, X_train.shape[0], self.batch_size):
            for j in range(i, i+self.batch_size):
                z = self.forward(X_train)
                self.backpropagation(z)

    def forward(self, X):
        z1 = self.convolution(X)
        z2 = self.ReLU(z1)
        z3 = self.max_pooling(z2)
        z4 = np.dot(z3, self.w) + self.b
        z5 = self.sigmoid(z4)
        y = self.softmax(z5)
        return y

    def convolution(self, X):
        pass

    def ReLU(self, X):
        pass

    def max_pooling(self):
        pass

    def sigmoid(self):
        pass

    def softmax(self):
        pass

    def predict(self, X):
        pass

    def backpropagation(self):
        pass

    def loss(self):
        # cross entropy loss
        pass

    def score(self, X, y):
        pass

    def save_model(self):
        pass
'''
class SimpleCNN:
    def __init__(self):
        pass

    def convolution(self):
        pass

    def max_pooling(self):
        pass

class Perceptron:
    def __init__(self):
        self.w = 0
        self.b = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, x_i, x):
        return np.exp(x_i) / sum([np.exp(t) for t in x])

    def sigma(self, X):
        return np.sum(X * self.w) + self.b

    def backdrop(self, y, a, X):
        for i in range(X.shape[0]):
            self.w[i] += (y - a) * X[i]
        self.b += y - a

    def loss(self):
        pass

    def fit(self, X, y, epochs=100, batch_size=32):
        self.w = np.ones(X.shape[0])
        self.b = 0
        for i in range(epochs):
            loss = 0
            for j in range(X.shape[0]):
                z = self.sigma(X[i])
                a = self.sigmoid(z)
                # update weight and bias
                self.backdrop(y, a)
                # a = np.clip(a, 1e-10, 1-1e-10)
                loss += [self.sigma(x_i) for x_i in X]
                return np.array(z) > 0


    def predict(self, x):
        a1 = np.dot(x, self.w) + self.b
        z1 = self.sigmoid(a1)
        a2 = np.dot(a1, self.w) + self.b
        z2 = self.sigmoid(a2)
        a3 = np.dot(a2, self.w) + self.b
        z3 = self.softmax(a3)
        return max(z3)

    def accuracy(self, X, y):
        X_predict = [self.predict(t) for t in X]
        result = list(map(lambda X_p, y: 1 if X_p==y else 0, X_predict, y))
        return sum(result) / len(X)

    def precision_score(self):
        pass

    def recall_score(self):
        pass

    def confusion_matrix(self):
        pass
'''