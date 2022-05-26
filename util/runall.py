
from adaboost import boost
from bayes import naive
from decisiontree import tree, forest
from knn import knn
from perceptron import perceptron
from regression import linear_regression, logistic_regression
from svm import svm


if __name__ == '__main__':
    verbose = False
    boost.run_tests(verbose=verbose)
    naive.run_tests(verbose=verbose)
    tree.run_tests(verbose=verbose)
    forest.run_tests(verbose=verbose)
    knn.run_tests(verbose=False)
    perceptron.run_tests(verbose=verbose)
    linear_regression.run_tests(verbose=verbose)
    logistic_regression.run_tests(verbose=verbose)
    svm.run_tests(verbose=verbose)

    # TODO: tests for lda, pca, and kmeans