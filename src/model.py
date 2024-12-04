import numpy as np
import pickle
from tqdm import tqdm


class Convolution:
    def __init__(self, filter_shape=(3, 1, 3, 3), padding=0, stride=1, activation=None):
        self.input_shape = None              # (C, H, W)
        self.filter_shape = filter_shape     # (FN, C, FH, FW)
        self.output_shape = None             # (FN, OH, OW)

        self.padding = padding
        self.stride = stride

        self.w = None                       # (FN, C, FH, FW)
        self.X = None
        self.col = None
        self.col_w = None

        self.activation = activation

    def set_parameters(self, input_shape):
        self.input_shape = input_shape
        C, H, W = input_shape
        FN, _, FH, FW = self.filter_shape
        self.output_shape = (FN, (H+2*self.padding-FW) // self.stride+1, (W+2*self.padding-FW) // self.stride+1)
        self.w = np.random.normal(loc=0, scale=2/(np.prod(self.input_shape)**0.5), size=(FN, C, FH, FW))
        return self.output_shape

    def im2col(self, X):
        N, C, H, W = X.shape
        FN, _, FH, FW = self.filter_shape
        OH, OW = self.output_shape[1:]
        after_padding = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)
        col = np.zeros((N, C, FH, FW, OH, OW))
        # (N, C, FH, FW, OH, OW)

        for y in range(FH):
            y_max = y + self.stride * OH
            for x in range(FW):
                x_max = x + self.stride * OW
                col[:, :, y, x, :, :] = after_padding[:, :, y:y_max:self.stride, x:x_max:self.stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
        return col

    def col2im(self, col):
        N, C, H, W = self.X.shape
        FN, _, FH, FW = self.filter_shape
        OH, OW = self.output_shape[1:]
        col = col.reshape(N, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)

        im = np.zeros((N, C, H+2*self.padding+self.stride-1, W+2*self.padding+self.stride-1))
        for y in range(FH):
            y_max = y + self.stride * OH
            for x in range(FW):
                x_max = x + self.stride * OW
                im[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]

        return im[:, :, self.padding:H+self.padding, self.padding:W+self.padding]

    def convolution(self, Z, N):
        self.col_w = self.w.reshape(self.filter_shape[0], -1).T
        output = np.dot(Z, self.col_w).reshape(N, *self.output_shape)
        return output

    def ReLU(self, Z):
        return np.maximum(0, Z)

    def forward(self, X):
        N = X.shape[0]
        self.X = X
        self.col = self.im2col(self.X)
        Z = self.convolution(self.col, N)
        if not self.activation:
            return Z
        elif self.activation == "relu":
            return self.ReLU(Z)

    def backward(self, grad, lr):
        FN, C, FH, FW = self.filter_shape
        N = grad.shape[0]
        grad = grad.transpose(0, 2, 3, 1).reshape(-1, FN)

        dw = np.dot(self.col.T, grad)
        dw = dw.transpose(1, 0).reshape(FN, C, FH, FW)
        self.w -= lr * dw / N

        dcol = np.dot(grad, self.col_w.T)
        dx = self.col2im(dcol)

        return dx


class MaxPooling:
    def __init__(self, PH=2, PW=2, stride=2):
        self.input_shape = None             # (C, H, W)
        self.output_shape = None            # (C, OH, OW)
        self.PH = PH
        self.PW = PW
        self.stride = stride

        self.max_arg = None

    def set_parameters(self, input_shape):
        self.input_shape = input_shape          # (C, H, W)
        C, H, W = input_shape
        # (FN, OH, OW)
        self.output_shape = (C, (H-self.PH)//self.stride+1, (W-self.PW)//self.stride+1)
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

    def col2im(self, col):
        N = col.shape[0]
        C, H, W = self.input_shape
        FN, OH, OW = self.output_shape
        PH, PW = self.PH, self.PW
        col = col.reshape(N, OH, OW, C, PH, PW).transpose(0, 3, 4, 5, 1, 2)

        im = np.zeros((N, C, H, W))
        for y in range(PH):
            y_max = y + self.stride * OH
            for x in range(PW):
                x_max = x + self.stride * OW
                im[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]

        return im

    def forward(self, X):
        N = X.shape[0]
        col = self.im2col(X).reshape(-1, self.PH * self.PW)
        out = np.max(col, axis=1).reshape(N, *self.output_shape)
        self.max_arg = np.argmax(col, axis=1).reshape(N, -1).reshape(N, self.output_shape[0], -1)
        return out

    def backward(self, grad, lr):
        N = grad.shape[0]
        C, OH, OW = self.output_shape
        grad = grad.reshape(N, C, OH, OW)
        pool_size = self.PH * self.PW
        dmax = np.zeros((grad.size, pool_size))
        dmax[np.arange(self.max_arg.size), self.max_arg.flatten()] = grad.flatten()
        dmax = dmax.reshape(grad.shape + (pool_size,))
        print('dmax', dmax.shape)
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1], dmax.shape[2], -1)
        dx = self.col2im(dcol)
        print(dcol.shape, dx.shape)
        return dx


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

    def set_parameters(self, input_shape):
        self.input_shape = input_shape
        range = (6 / (np.prod(self.input_shape) * np.prod(self.output_shape))) ** 0.5
        self.w = np.random.uniform(low=-range, high=range, size=(np.prod(self.input_shape), self.output_shape[1]))
        self.b = np.random.uniform(low=-range, high=range, size=(1, self.output_shape[1]))
        return self.output_shape

    def sigma(self, X):
        return np.dot(X, self.w) + self.b

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        z = np.nan_to_num(z, nan=0.0, posinf=1e7, neginf=1e7)
        z_normalized = z - np.max(z, axis=1, keepdims=True)
        self.softmax_sum = np.sum(np.exp(z_normalized), axis=1, keepdims=True)
        return np.exp(z_normalized) / self.softmax_sum

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
        N = grad.shape[0]
        if self.activation == 'softmax':
            dL_dZ = grad - np.sum(grad*self.softmax(self.Z), axis=1, keepdims=True)
        elif self.activation == 'sigmoid':
            pass
        else:   # activation = None
            dL_dZ = grad
        dL_dw = np.dot(self.X.T, dL_dZ)
        dL_dX = np.dot(dL_dZ, self.w.T).reshape(-1, *self.input_shape)
        dL_db = np.sum(dL_dZ, axis=0, keepdims=True)
        self.w -= lr * dL_dw / N
        self.b -= lr * dL_db / N
        return dL_dX


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
        # bound = (6 / (input_shape[0] * input_shape[1] + output_shape[1])) ** 0.5        # Glorot initialization
        self.input_shape = input_shape
        self.output_shape = output_shape
        for layer in self.layers:
            output_shape = layer.set_parameters(input_shape)
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
        result = self.forward(X)
        idx = np.argmax(result, axis=1)
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
        pred = self.predict(X, one_hot_encoding=False)
        true_labels = np.argmax(Y, axis=1)
        return np.mean(pred == true_labels)

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
