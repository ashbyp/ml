from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.naive_bayes import GaussianNB
from util.common import accuracy, class_name, load_spam


class NumpyNaiveBayes:

    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for c in self.classes:
            X_c = X[c == y]
            self.mean[c,:] = X_c.mean(axis=0)
            self.var[c, :] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0]/float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self.probability_density(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def probability_density(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


def test_bayes(bayes, dataset_name, X, y, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    if verbose:
        print(f'X_test shape {X_test.shape}')
        print(f'y_test shape {X_test.shape}')

    bayes.fit(X_train, y_train)
    predictions = bayes.predict(X_test)

    if verbose:
        print(f'Actual : {y_test}')
        print(f'Predict: {predictions}')

    print(f'{class_name(bayes)} accuracy with dataset {dataset_name} {accuracy(y_test, predictions)}')


def run_tests(verbose=False):
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)

    test_bayes(NumpyNaiveBayes(), "classification", X, y, verbose)
    test_bayes(GaussianNB(), "classification", X, y, verbose)


if __name__ == '__main__':
    run_tests(verbose=False)