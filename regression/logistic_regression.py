import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression as SK
from sklearn.model_selection import train_test_split

from util.common import run_test_with_accuracy
from util.data import load_uci


class NumpyLogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


def test(lr, X, y, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)

    if verbose:
        print(f'Actual : {y_test}')
        print(f'Predict: {predictions}')

    return y_test, predictions


def run_tests(verbose=False):
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    run_test_with_accuracy(test, NumpyLogisticRegression(lr=0.0001), "breast cancer", X, y, verbose)
    run_test_with_accuracy(test, SK(random_state=0, max_iter=10000), "breast cancer", X, y, verbose)

    for uci in ('spam', 'SPECT heart', 'wine'):
        X, y = load_uci(uci, verbose=verbose)
        run_test_with_accuracy(test, NumpyLogisticRegression(lr=0.0001), uci, X, y, verbose)
        run_test_with_accuracy(test, SK(random_state=0, max_iter=10000), uci, X, y, verbose)


if __name__ == '__main__':
    run_tests(verbose=False)
