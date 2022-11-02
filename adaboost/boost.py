from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from loading import loaddata


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class Boost:

    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []

        # Iterate through classifiers
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float("inf")

            # greedy search to find best threshold and feature
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    # predict with polarity 1
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Error = sum of weights of misclassified samples
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # store the best configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            # calculate alpha
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # calculate predictions and update weights
            predictions = clf.predict(X)

            w *= np.exp(-clf.alpha * y * predictions)
            # Normalize to one
            w /= np.sum(w)

            # Save classifier
            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred


def test_bc(b, pr=False):
    def accuracy(y_true, y_pred):
        acc = np.sum(y_true == y_pred) / len(y_true)
        return acc

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    b.fit(X_train, y_train)
    predictions = b.predict(X_test)

    if pr:
        print(f'Actual : {y_test}')
        print(f'Predict: {predictions}')

    print(f'Accuracy {accuracy(y_test, predictions)}')


def test_spam(b, pr=False):
    def accuracy(y_true, y_pred):
        acc = np.sum(y_true == y_pred) / len(y_true)
        return acc

    loaddata.download_spam()
    data = loaddata.open_with_np()
    print(f'CSV data shape: {data.shape}')
    X, y = loaddata.split_samples_and_features(data)
    print(f'X: {X.shape}')
    print(f'y: {y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    b.fit(X_train, y_train)
    predictions = b.predict(X_test)

    if pr:
        print(f'Actual : {y_test}')
        print(f'Predict: {predictions}')

    print(f'Accuracy {accuracy(y_test, predictions)}')


if __name__ == '__main__':
    test_bc(Boost())
    test_spam(Boost())

    print('=' * 80)

    from sklearn.ensemble import AdaBoostClassifier

    test_bc(AdaBoostClassifier())
    test_spam(AdaBoostClassifier())







