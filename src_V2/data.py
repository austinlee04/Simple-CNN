import numpy as np
import os

def load_data(path, one_hot_encoding=True):
    image_source = {
        "train_image": "train-images.idx3-ubyte",
        "test_image": "t10k-images.idx3-ubyte",
    }
    label_source = {
        "train_label": "train-labels.idx1-ubyte",
        "test_label": "t10k-labels.idx1-ubyte"
    }
    MNIST_data = {}
    for key in image_source:
        with open(os.path.join(os.path.dirname(__file__), '..', path, image_source[key]), 'rb') as f:
            MNIST_data[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28) / 255
    for key in label_source:
        with open(os.path.join(os.path.dirname(__file__), '..', path, label_source[key]), 'rb') as f:
            MNIST_data[key] = np.frombuffer(f.read(), np.uint8, offset=8)
        if one_hot_encoding:
            num = np.unique(MNIST_data[key], axis=0).shape[0]
            MNIST_data[key] = np.eye(num)[MNIST_data[key]]
    return MNIST_data["train_image"], MNIST_data["test_image"], MNIST_data["train_label"], MNIST_data["test_label"]
