import numpy as np


def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc


def class_name(cls, width=25):
    return f'{type(cls).__name__:{width}s}'


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def mean_square_error(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)