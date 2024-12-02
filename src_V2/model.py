import numpy as np
import pickle
from tqdm import tqdm


class CNN:
    def __init__(self, filter_shape):
        self.input_shape = None             # (C, H, W)
        self.filter_shape = filter_shape    # (FN, FH, FW)
        self.output_shape = None            # (OH, OW)
        self.padding, self.stride = 0, 0

    def im2col(self, X):
        N, C, H, W = X.shape
        FN, C, FH, FW = self.filter_shape
        _, OH, OW = self.output_shape
        col = np.zeros((N, C, FH, FW, OH, OW))
        print(col.shape)
        for y in range(FH):
            y_max = y + self.stride + OH
            for x in range(FW):
                x_max = x + self.stride + OW
                col[:, :, y, x, :, :] = X[:, :, y:y_max:self.stride, x:x_max:self.stride]
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*OH*OW, -1)
        return col

    def col2im(self, col):
        N = col.shape[0]
        C, H, W = self.input_shape
        FN, FH, FW = self.filter_shape
        OH, OW = self.filter_shape
        col = col.reshape(N, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)
        im = np.zeros((N, C, H+2*self.padding+self.stride-1, W+2*self.padding+self.stride-1))
        for y in range(FH):
            y_max = y + self.stride * OH
            for x in range(FW):
                x_max = x + self.stride * OW
                im[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]
        return im[:, :, self.padding:H+self.padding, self.padding:W+self.padding]


class Convolution(CNN):
    def __init__(self, filter_shape, padding, stride, activation):
        super().__init__(filter_shape)
        self.padding = padding
        self.stride = stride
        self.activation = activation

        self.X = None
        self.w = None
        self.col = None
        self.w_col = None

    def set_parameters(self, input_shape):
        self.input_shape = input_shape
        C, H, W = self.input_shape
        FN, _, FH, FW = self.filter_shape
        self.output_shape = (FN, (H+2*self.padding-FH)//self.stride+1, (W+2*self.padding-FW)//self.stride+1)
        # He initialization
        self.w = np.random.normal(loc=0, scale=2/np.prod(self.input_shape), size=(FN, C, FH, FW))
        return self.output_shape

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    def forward(self, X):
        N = X.shape[0]
        self.X = X
        FN = self.filter_shape[0]
        self.col = self.im2col(X)
        self.w_col = self.w.reshape(FN, -1).T
        Z = np.dot(X, self.w_col).reshape(N, *self.output_shape)
        if not self.activation:
            return Z
        elif self.activation == 'relu':
            return self.relu(Z)

    def backward(self, grad, lr):
        N = grad.shape[0]
        FN, C = self.filter_shape[:2]

        grad = grad.transpose(0, 2, 3, 1).reshape(-1, FN)
        dw = np.dot(self.col.T, grad).T.reshape(self.filter_shape)      # (FN, C, FH, FW)
        self.w -= lr * dw / N

        dcol = np.dot(grad, self.w_col.T)
        dx = self.col2im(dcol)
        return dx


class MaxPooling(CNN):
    def __init__(self, filter_shape, stride):
        super().__init__(filter_shape)
        self.stride = stride
        self.max_arg = None

    def set_parameters(self, input_shape):
        self.input_shape = input_shape
        C, H, W = input_shape
        FH, FW = self.filter_shape
        self.filter_shape = (C, FH, FW)
        self.output_shape = (C, (H-FH)//self.stride+1, (W-FW)//self.stride+1)       # (C, OH, OW)
        return self.output_shape

    def forward(self, X):
        N = X.shape[0]
        FN, FH, FW = self.filter_shape
        C, OH, OW = self.output_shape
        col = self.im2col(X).reshape(-1, FH*FW)
        out = np.max(col, axis=1).reshape(N, C, OH, OW)
        self.max_arg = np.argmax(col, axis=1).reshape(N, C, -1)
        return out

    def backward(self, grad, lr):
        N = grad.shape[0]
        C, FH, FW = self.filter_shape
        _, OH, OW = self.output_shape
        grad = grad.reshape(N, C, OH, OW)
        pool_size = FH * FW
        d_max = np.zeros((grad.size, pool_size))
        d_max[np.arange(self.max_arg.size), self.max_arg.flatten()] = grad.flatten()
        d_max = d_max.reshape(grad.shape + (pool_size,))
        d_col = d_max.reshape(d_max.shape[0]*d_max.shape[1], d_max.shape[2], -1)
        dx = self.col2im(d_col)
        return dx


class Affine:
    def __init__(self):
        self.input_shape = None
        self.num_classes = 10
        self.w = None
        self.b = None
        self.X = None
        self.Z = None

    def set_parameters(self, input_shape):
        self.input_shape = input_shape
        range = (6 / (np.prod(self.input_shape) + self.num_classes)) ** 0.5
        self.w = np.random.uniform(low=-range, high=range, size=(np.prod(self.input_shape), self.num_classes))
        self.b = np.random.uniform(low=-range, high=range, size=self.num_classes)
        return self.num_classes

    def softmax(self, X):
        X_normalized = X - np.max(X, axis=1, keepdims=True)
        return np.exp(X_normalized) / np.sum(np.exp(X_normalized), axis=1, keepdims=True)

    def forward(self, X):
        N = X.shape[0]
        self.X = X.reshape(N, -1)
        self.Z = np.dot(self.X, self.w) + self.b
        Y = self.softmax(self.Z)
        return Y

    def backward(self, grad, lr):
        N = grad.shape[0]
        dL_dZ = grad * self.softmax(self.Z)
        dL_dw = np.dot(self.X.T, dL_dZ)
        dL_db = np.sum(dL_dZ, axis=0, keepdims=True)
        self.w -= lr * dL_dw / N
        self.b -= lr * dL_db / N
        dL_dX = np.dot(dL_dZ, self.w.T).reshape(-1, *self.input_shape)
        return dL_dX


class Model:
    def __init__(self):
        self.learning_rate = 1
        self.layers = []
        self.input_shape = (1, 28, 28)
        self.num_class = 10

    def add_layers(self, *models):
        self.layers.extend(models)

    def set_parameters(self):
        input_shape = self.input_shape
        for layer in self.layers:
            output_shape = layer.set_parameters(input_shape)
            input_shape = output_shape

    @staticmethod
    def cross_entropy_loss(pred, ans):
        loss = -1 * np.log(pred + 1e-7) * ans
        loss = np.sum(loss)
        return loss

    def forward(self, X):
        X = X.reshape(-1, *self.input_shape)        # (N, C, H, W)
        N = X.shape[0]
        for layer in self.layers:
            X = layer.forward(X)
        return X.reshape(N, -1)                     # (N, num_classes)

    def backward(self, Y_pred, Y_ans):
        N = Y_pred.shape[0]
        grad = -1 * Y_ans / (Y_pred * 1e-7)
        for i in range(len(self.layers)-1, -1, -1):
            grad = self.layers[i].backward(grad, self.learning_rate)

    def predict(self, X):
        N = X.shape[0]
        result = self.forward(X)
        pred = np.argmax(result, axis=1)
        return pred

    def score(self, X, Y):
        result = self.predict(X)
        true_labels = np.argmax(Y, axis=1)
        return np.mean(result == true_labels)

    def fit(self, X_train, y_train, X_test, y_test, epochs=500, batch_size=100, learning_rate=0.01, early_stopping=False):
        self.learning_rate = learning_rate
        for epoch in range(1, epochs+1):
            loss = 0
            for i in tqdm(range(0, X_train.shape[0], batch_size), desc=f"epoch {epoch}/{epochs}"):
                Y = self.forward(X_train[i:i+batch_size])
                loss += np.sum(self.cross_entropy_loss(Y, y_train[i:i+batch_size]))
                self.backward(Y, y_train[i:i+batch_size])
            print(f"\ntraining in process (epoch {epoch}/{epochs}) : loss {loss} | train score {self.score(X_train, y_train)} | test score {self.score(X_test, y_test)}")
            if early_stopping:
                pass
