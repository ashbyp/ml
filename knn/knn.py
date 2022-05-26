from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from util.common import euclidean_distance, class_name, accuracy, load_spam


def plot(X, y):
    colormap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap, edgecolor='k', s=20)
    plt.show()


class NumpyKNN:

    def __init__(self, k=3):
        self.X_train = self.y_train = None
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def test_knn(knn, dataset_name, X, y, verbose=False):
    if verbose:
        plot(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    if verbose:
        print(f'Shape of training data: {X_train.shape}')
        print(f'First row of training data is: {X_train[0]}')
        print(f'Shape of target data: {y_train.shape}')
        print(f'First row of target data is: {y_train[0]}')

    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    if verbose:
        print(predictions)
        print(y_test)

    print(f'{class_name(knn)} accuracy with dataset {dataset_name} {accuracy(y_test, predictions)}')


def run_tests(verbose=False):
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    test_knn(NumpyKNN(k=3), "iris", X, y, verbose)
    test_knn(KNeighborsClassifier(n_neighbors=3), "iris", X, y, verbose)

    X, y = load_spam()
    test_knn(NumpyKNN(k=3), "spam", X, y, verbose)
    test_knn(KNeighborsClassifier(n_neighbors=3), "spam", X, y, verbose)


if __name__ == '__main__':
    run_tests(verbose=False)

