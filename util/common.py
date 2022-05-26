import numpy as np
from loading import loaddata


def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc


def class_name(cls):
    return type(cls).__name__


def load_spam():
    """Returns spam features,target in the right format to work with"""
    filename, _ = loaddata.download_spam(force_download=False)
    data = loaddata.open_with_np(filename)
    return loaddata.split_features_and_labels(data)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def mean_square_error(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)