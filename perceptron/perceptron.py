import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

from util.common import run_test_with_accuracy


class NumpyPerceptron:

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear)
                update = self.lr * (y_[idx]-y_predicted)
                self.weights += update * x_i
                self.bias += update

        return None

    def predict(self, x):
        linear = np.dot(x, self.weights) + self.bias
        y_predicted = self.activation_func(linear)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x>0, 1, 0)


def test(perceptron, X, y, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    perceptron.fit(X_train, y_train)
    predictions = perceptron.predict(X_test)

    if verbose:
        print(predictions)
        print(y_test)

    if verbose and hasattr(perceptron, 'weights'):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

        x0_1 = np.amin(X_train[:, 0])
        x0_2 = np.amax(X_train[:, 0])

        x1_1 = (-perceptron.weights[0] * x0_1 - perceptron.bias) / perceptron.weights[1]
        x1_2 = (-perceptron.weights[0] * x0_2 - perceptron.bias) / perceptron.weights[1]

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

        ymin = np.amin(X_train[:, 1])
        ymax = np.amax(X_train[:, 1])
        ax.set_ylim([ymin - 3, ymax + 3])

        plt.show()

    return y_test, predictions


def run_tests(verbose=False):
    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)

    run_test_with_accuracy(test, NumpyPerceptron(), "blobs", X, y, verbose)
    run_test_with_accuracy(test, Perceptron(), "blobs", X, y, verbose)


if __name__ == '__main__':
    run_tests(verbose=True)



