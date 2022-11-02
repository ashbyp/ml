import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def plot(X, y):
    colormap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap, edgecolor='k', s=20)
    plt.show()


class KNN:

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


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    plot(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    print(f'Shape of training data: {X_train.shape}')
    print(f'First row of training data is: {X_train[0]}')
    print(f'Shape of target data: {y_train.shape}')
    print(f'First row of target data is: {y_train[0]}')

    k = KNN(k=3)
    k.fit(X_train, y_train)
    predictions = k.predict(X_test)

    print(predictions)
    print(y_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(accuracy)

    print('=' * 80)

    # Using sklearn
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    predictions = neigh.predict(X_test)
    print(predictions)
    print(y_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(accuracy)
