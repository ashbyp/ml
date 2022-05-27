import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import sklearn.svm
from sklearn.model_selection import train_test_split

from util.common import class_name, accuracy


class NumpySupportVectorMachine:

    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                cond = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if cond:
                    self.weights -= self.lr * (2* self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2* self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias =- self.lr * y_[idx]

    def predict(self, x):
        linear = np.dot(x, self.weights) - self.bias
        return np.sign(linear)

    def visualize_svm(self, X, y):
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, self.weights, self.bias, 0)
        x1_2 = get_hyperplane_value(x0_2, self.weights, self.bias, 0)

        x1_1_m = get_hyperplane_value(x0_1, self.weights, self.bias, -1)
        x1_2_m = get_hyperplane_value(x0_2, self.weights, self.bias, -1)

        x1_1_p = get_hyperplane_value(x0_1, self.weights, self.bias, 1)
        x1_2_p = get_hyperplane_value(x0_2, self.weights, self.bias, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()


def test_svm(svm, dataset_name, X, y, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)

    if verbose:
        print(predictions)
        print(y_test)

    print(f'{class_name(svm)} accuracy with dataset {dataset_name} {accuracy(y_test, predictions)}')

    if verbose and hasattr(svm, "visualize_svm"):
        svm.visualize_svm(X, y)


def run_tests(verbose=False):
    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    y = np.where(y == 0, -1, 1)

    test_svm(NumpySupportVectorMachine(), "blobs", X, y, verbose)
    test_svm(sklearn.svm.SVC(), "blobs", X, y, verbose)


if __name__ == '__main__':
    run_tests(verbose=True)



