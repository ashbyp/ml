from collections import Counter

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from decisiontree.tree import NumpyDecisionTree
from util.common import run_test_with_accuracy


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]


class NumpyRandomForest:

    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = NumpyDecisionTree(self.min_samples_split, self.max_depth, self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self._most_common_label(tree_pred) for tree_pred in tree_preds]
        return y_pred

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


def test(forest, X, y, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    forest.fit(X_train, y_train)
    predictions = forest.predict(X_test)

    if verbose:
        print(f'Actual : {y_test}')
        print(f'Predict: {predictions}')

    return y_test, predictions


def run_tests(verbose=False):
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    run_test_with_accuracy(test, NumpyRandomForest(n_trees=3, max_depth=10), 'breast cancer', X, y, verbose)


if __name__ == '__main__':
    run_tests(verbose=False)