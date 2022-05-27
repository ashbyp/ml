from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression as SK
from util.common import mean_square_error, class_name


class NumpyLinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # gradient descent method
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        print(f'Weights: {self.weights}')
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) +  self.bias
        return y_predicted


def test(lr, dataset_name, X, y, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    if verbose:
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
        plt.show()

        print(X_test)
        print(y_test)
        print(X_test.shape)
        print(y_test.shape)

    lr.fit(X_train, y_train)
    predicted = lr.predict(X_test)

    print(f'{class_name(lr)} MSE with dataset {dataset_name} {mean_square_error(y_test, predicted)}')

    if verbose:
        y_pred_line = lr.predict(X)
        cmap = plt.get_cmap("viridis")
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
        plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
        plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
        plt.show()


def run_tests(verbose=False):
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    test(NumpyLinearRegression(lr=0.01), "regression", X, y, verbose)
    test(SK(), "regression", X, y, verbose)


if __name__ == '__main__':
    run_tests(verbose=False)

