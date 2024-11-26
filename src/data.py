import numpy as np
import os

def load_data(path, normalize=True):
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
            MNIST_data[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
            if normalize: MNIST_data[key] /= 255
    for key in label_source:
        with open(os.path.join(os.path.dirname(__file__), '..', path, label_source[key]), 'rb') as f:
            MNIST_data[key] = np.frombuffer(f.read(), np.uint8, offset=8)
    return MNIST_data["train_image"], MNIST_data["test_image"], MNIST_data["train_label"], MNIST_data["test_label"]
