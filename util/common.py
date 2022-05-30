import numpy as np
import timeit


def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc


def class_name(cls, width=25):
    return f'{type(cls).__name__:{width}s}'


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def mean_square_error(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)


def run_test_with_accuracy(testfn, algo, dataset_name, X, y, verbose, accuracyfn=accuracy):
    start = timeit.default_timer()
    actual, predict = testfn(algo, X, y, verbose)
    elapsed = timeit.default_timer() - start
    print(f'{class_name(algo)} {accuracyfn.__name__} with dataset {dataset_name} {accuracyfn(actual, predict):4f} in {elapsed:.4f}s')
