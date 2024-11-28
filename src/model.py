import numpy as np
import pickle


class Convolution:
    def __init__(self, input_size, filter_size, padding=0, stride=1, activation='ReLU'):
        self.input_size = input_size
        self.filter_size = filter_size          # (3,3)
        self.output_size = 0

        self.padding = padding
        self.stride = stride

        self.activation = activation

    def im2col(self):
        pass

    def convolution(self):
        pass

    def ReLU(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class Pooling:
    def __init__(self, pooling_type="max_pooling"):
        self.pooling_type = pooling_type

    def max_pooling(self):
        pass

    def avg_pooling(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class FullyConnectedLayer:
    def __init__(self, use_softmax=False):
        self.w = 0
        self.b = 0
        self.use_softmax = use_softmax

    @staticmethod
    def sigma(self, X):
        return np.dot(X, self.w) + self.b

    @staticmethod
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(self, z):
        Z = sum([np.exp(t) for t in z])
        return np.exp(z) / Z

    def forward(self, X):
        Z1 = self.sigma(X)
        Z2 = self.sigmoid(Z1)
        if self.use_softmax:
            return self.softmax(Z2)
        else:
            return Z2

    def backward(self):
        pass


class SimpleCNN:
    def __init__(self, learning_rate, input_size):
        self.lr = learning_rate
        self.layers = []
        self.input_size = input_size

    def loss(self):
        # cross-entropy loss
        pass

    def add_layer(self, model):
        self.layers.append(model)

    def forward(self, X):
        if len(self.layers) == 0:
            print("Error : no layer added to model")
            return False
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dL_dw):
        if len(self.layers) == 0:
            print("Error : no layer added to model")
            return False
        for layer in self.layers[::-1]:
            pass

    def fit(self, X_train, y_train, epochs=500, batch_size=128, early_stopping=False, validation=False):
        for i in range(0, X_train.shape[0], self.batch_size):
            for j in range(i, i+self.batch_size):
                pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass
'''
class Perceptron:

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