import pickle
import os
from data import load_data

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), '../ckpt/ckpt.pkl'), 'rb') as fr:
        model = pickle.load(fr)
    X_train, X_test, y_train, y_test = load_data("./dataset")

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"train score : {train_score} | test score : {test_score}")

