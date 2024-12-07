import numpy as np
from data import load_data
from model import Model, Convolution, MaxPooling, Affine
import pickle
import matplotlib.pyplot as plt


def train(epochs, batch_size, learning_rate, early_stopping=False):
    X_train, X_test, y_train, y_test = load_data("./dataset")
    model = Model()
    # add layers to the model
    model.add_layers(Convolution(filter_shape=(5, 1, 3, 3), padding=1, stride=1, activation='relu'),
                     MaxPooling((2, 2), stride=2),
                     Convolution(filter_shape=(1, 5, 3, 3), padding=1, stride=1, activation="relu"),
                     MaxPooling((2, 2), stride=2),
                     Affine())
    model.set_parameters()
    train_scores, test_scores, losses = [], [], []
    for epoch in range(1, epochs+1):
        # apply learning rate decay for faster learning
        lr = learning_rate/(epoch**0.75)
        loss = model.fit(X_train, y_train, batch_size=batch_size, learning_rate=lr)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"\ntraining in process (epoch {epoch}/{epochs}, learning rate {lr}) : train loss {loss} | train score {train_score} | test score {test_score}")
        train_scores.append(train_score)
        test_scores.append(test_score)
        losses.append(loss)

        if early_stopping:
            pass

    with open('../ckpt/ckpt.pkl', 'wb') as f:
        pickle.dump(model, f)

    idx = np.arange(1, epochs+1)
    plt.plot(idx, train_scores, color='blue', label='train score')
    plt.plot(idx, test_scores, color='orange', label='test score')
    plt.plot(idx, losses, color='green', label='train loss')
    plt.xlabel('epochs')
    plt.ylabel('accuracy or loss')
    plt.title('Model Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train(epochs=10, batch_size=64, learning_rate=0.01)
