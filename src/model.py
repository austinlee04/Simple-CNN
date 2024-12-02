import numpy as np
import pickle
from tqdm import tqdm


class Convolution:
    def __init__(self, filter_shape=(3, 1, 3, 3), padding=0, stride=1, activation=None):
        self.input_shape = None              # (C, H, W)
        self.filter_shape = filter_shape     # (FN, C, FH, FW), C=1
        self.output_shape = None             # (FN, OH, OW)

        self.padding = padding
        self.stride = stride

        self.w = None                       # (FN, C, FH, FW)

        self.activation = activation

        self.X = None

    def set_parameters(self, input_shape, bound):
        self.input_shape = input_shape
        self.w = np.random.uniform(-bound, bound, self.filter_shape)
        self.output_shape = (self.filter_shape[0],
                             (self.input_shape[1] + 2 * self.padding - self.filter_shape[2]) // self.stride + 1,
                             (self.input_shape[2] + 2 * self.padding - self.filter_shape[3]) // self.stride + 1)
        return self.output_shape

    def im2col(self, X):
        N = X.shape[0]
        after_padding = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)
        col = np.zeros((N, self.input_shape[0], self.filter_shape[2], self.filter_shape[3], self.output_shape[1], self.output_shape[2]))
        # (N, C, FH, FW, OH, OW)

        for y in range(self.filter_shape[2]):
            y_max = y + self.stride * self.output_shape[1]
            for x in range(self.filter_shape[3]):
                x_max = x + self.stride * self.output_shape[2]
                col[:, :, y, x, :, :] = after_padding[:, :, y:y_max:self.stride, x:x_max:self.stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * self.output_shape[1] * self.output_shape[2], -1)

        return col

    def convolution(self, Z, N):
        col_filter = self.w.reshape(self.filter_shape[0], -1).T
        output = np.dot(Z, col_filter).reshape(N, *self.output_shape)
        return output

    def ReLU(self, Z):
        return np.maximum(0, Z)

    def forward(self, X):
        N = X.shape[0]
        self.X = X
        Z1 = self.im2col(self.X)
        Z2 = self.convolution(Z1, N)
        if not self.activation:
            return Z2
        elif self.activation == "relu":
            return self.ReLU(Z2)

    def backward(self, grad, lr):
        w_update = np.zeros_like(self.w)
        N = grad.shape[0]
        dL_dX = np.zeros_like(self.X)
        for i in range(0, self.output_shape[1]-self.filter_shape[2]+1, self.stride):
            for j in range(0, self.output_shape[2]-self.filter_shape[3]+1, self.stride):
                pass
            # dL_dX +=
        for i in range(N):
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[2]):
                    for fn in range(self.filter_shape[0]):
                        w_update += self.X[i, :, j*self.stride:j*self.stride+self.filter_shape[2],
                                    k*self.stride:k*self.stride+self.filter_shape[3]] * grad[fn, :, j, k]
        self.w -= lr * w_update / N
        return dL_dX


class MaxPooling:
    def __init__(self, PH=2, PW=2, stride=2):
        self.input_shape = None             # (C, H, W)
        self.output_shape = None            # (C, OH, OW)
        self.PH = PH
        self.PW = PW
        self.stride = stride

        self.max_arg = None

    def set_parameters(self, input_shape, bound):
        self.input_shape = input_shape          # (C, H, W)
        # (FN, OH, OW)
        self.output_shape = (self.input_shape[0],
                             int(1 + (self.input_shape[1] - self.PH) / self.stride),
                             int(1 + (self.input_shape[2] - self.PW) / self.stride))
        return self.output_shape

    def im2col(self, X):
        N = X.shape[0]

        col = np.zeros((N, self.input_shape[0], self.PH, self.PW, self.output_shape[1], self.output_shape[2]))      # (N, C, PH, PW, OH, OW)

        for y in range(self.PH):
            y_max = y + self.stride * self.output_shape[1]
            for x in range(self.PW):
                x_max = x + self.stride * self.output_shape[2]
                col[:, :, y, x, :, :] = X[:, :, y:y_max:self.stride, x:x_max:self.stride]
        # (N, C, PH, PW, OH, OW) -> (N, OH, OW, C, PH, PW) -> (N*OH*OW, -1)
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * self.output_shape[1] * self.output_shape[2], -1)
        return col

    def forward(self, X):
        N = X.shape[0]
        col = self.im2col(X).reshape(-1, self.PH * self.PW)
        out = np.max(col, axis=1).reshape(N, *self.output_shape)
        self.max_arg = np.argmax(col, axis=1).reshape(N, -1).reshape(N, self.output_shape[0], -1)
        return out

    def backward(self, grad, lr):
        # including implementation of column to image
        N = grad.shape[0]
        grad = grad.reshape(N, self.output_shape[0], -1)
        im = np.zeros((N, *self.input_shape))
        col_grad = np.zeros((N, self.output_shape[0], self.output_shape[1]*self.output_shape[2], self.PH * self.PW))
        for i in range(N):
            for j in range(self.max_arg.shape[2]):
                col_grad[i, :, j, self.max_arg[i, :, j]] = grad[i, :, j]
        col_grad = col_grad.reshape(N, self.output_shape[0], self.output_shape[1]*self.output_shape[2], self.PH, self.PW)
        idx = 0
        for i in range(0, self.input_shape[1], self.PH):
            for j in range(0, self.input_shape[2], self.PW):
                im[:, :, i:i+self.PH, j:j+self.PW] = col_grad[:, :, idx]
                idx += 1
        return im


class FullyConnectedLayer:
    def __init__(self, activation=None, output_shape=(1, 10)):
        self.input_shape = None
        self.output_shape = output_shape
        self.w = None
        self.b = None
        self.activation = activation
        self.softmax_sum = 1
        self.X = None
        self.Z = None

    def set_parameters(self, input_shape, bound):
        self.input_shape = input_shape
        self.w = np.random.uniform(-bound, bound, (self.input_shape[1]*self.input_shape[2], self.output_shape[1]))
        self.b = np.random.uniform(-bound, bound, (1, self.output_shape[1]))
        return self.output_shape

    def sigma(self, X):
        return np.dot(X, self.w) + self.b

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        z_normalized = z - np.max(z, axis=1, keepdims=True)
        self.softmax_sum = np.sum(np.exp(z_normalized), axis=1)
        return np.exp(z_normalized) / self.softmax_sum[:, np.newaxis]

    def forward(self, X):
        N = X.shape[0]
        self.X = X.reshape(N, -1)
        self.Z = self.sigma(self.X)
        if not self.activation :
            return self.Z
        elif self.activation == 'softmax':
            return self.softmax(self.Z)
        elif self.activation == 'sigmoid':
            return self.sigmoid(self.Z)

    def backward(self, grad, lr):
        if self.activation == 'softmax':
            self.Z -= np.max(self.Z, axis=1, keepdims=True)
            dL_dZ = np.exp(self.Z) * grad * (1/self.softmax_sum)[:, np.newaxis]
            dL_dw = np.dot(self.X.T, dL_dZ)
            dL_dX = np.dot(dL_dZ, self.w.T)
            self.w -= lr * dL_dw
            self.b -= lr * np.average(dL_dZ, axis=0)
            return dL_dX
        elif self.activation == 'sigmoid':
            pass
        else:   # activation = None
            pass


class SimpleCNN:
    def __init__(self):
        self.learning_rate = 1
        self.layers = []
        self.input_shape = None
        self.output_shape = None

    def add_layers(self, *models):
        for model in models:
            self.layers.append(model)

    def set_parameters(self, input_shape=(1, 28, 28), output_shape=(1, 10)):
        bound = (6 / (input_shape[0] * input_shape[1] + output_shape[1])) ** 0.5        # Glorot initialization
        self.input_shape = input_shape
        self.output_shape = output_shape
        for layer in self.layers:
            output_shape = layer.set_parameters(input_shape, bound)
            input_shape = output_shape

    @staticmethod
    def cross_entropy_loss(y_pred, y_ans):
        ce_loss = -1 * np.log(y_pred + 1e-7) * y_ans
        return np.sum(ce_loss)

    def forward(self, X):
        X = X.reshape(-1, *self.input_shape)
        N = X.shape[0]
        if len(self.layers) == 0:
            print("Error : no layer added to model")
            return False
        for layer in self.layers:
            X = layer.forward(X)
        return X.reshape(N, -1)

    def backward(self, Y_pred, Y_ans):
        N = Y_pred.shape[0]
        grad = -1 * Y_ans / (Y_pred+1e-7)
        for i in range(len(self.layers)-1, -1, -1):
            grad = self.layers[i].backward(grad, self.learning_rate)

    def predict(self, X, one_hot_encoding=True):
        N = X.shape[0]
        idx = np.argmax(self.forward(X), axis=1)
        if one_hot_encoding:
            pred = np.zeros((N, self.output_shape[1]))
            for i in range(N):
                pred[i, idx[i]] = 1
            return pred
        return idx

    def fit(self, X_train, y_train, epochs=500, batch_size=64, learning_rate=0.001, validation=False, val_size=0.2, early_stopping=False):
        self.learning_rate = learning_rate
        if validation:
            X_train, X_val, y_train, y_val = \
                X_train[X_train.shape[0] * val_size:], X_train[:X_train.shape[0] * val_size], \
                y_train[y_train.shape[0] * val_size:], y_train[:y_train.shape[0] * val_size]
        for epoch in range(1, epochs+1):
            loss = 0
            for i in tqdm(range(0, X_train.shape[0], batch_size), desc=f"epoch {epoch}/{epochs}"):
                Y = self.forward(X_train[i:i+batch_size])
                loss += np.sum(self.cross_entropy_loss(Y, y_train[i:i + batch_size]))
                self.backward(Y, y_train[i:i+batch_size])
                # print(f"training in process (epoch {epoch}/{epochs}, batch {i//batch_size}/{X_train.shape[0]//batch_size})")
            print(f"\ntraining in process (epoch {epoch}/{epochs}) : loss {loss} | "
                  f"train score {self.score(X_train, y_train)}", end='')
            if validation:
                print(f" | validation score {self.score(X_val, y_val)}", end='')
            print()

    def score(self, X, Y):
        X_predict = self.predict(X)
        result = 0
        for pred, ans in zip(X_predict, Y):
            if np.argmax(pred) == np.argmax(ans):
                result += 1
        return result / len(X)

    def precision_score(self):
        pass

    def recall_score(self):
        pass

    def confusion_matrix(self):
        pass

    def roc_auc_score(self):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass
