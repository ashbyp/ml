from adaboost import boost
from bayes import naive
from decisiontree import tree, forest
from kmeans import kmeans
from knn import knn
from perceptron import perceptron
from regression import linear_regression, logistic_regression
from svm import svm

if __name__ == '__main__':
    verbose = False


    def _r(fn):
        print('-' * 80)
        fn(verbose)


    _r(boost.run_tests)
    _r(naive.run_tests)
    _r(tree.run_tests)
    _r(forest.run_tests)
    _r(kmeans.run_tests)
    _r(knn.run_tests)
    _r(perceptron.run_tests)
    _r(linear_regression.run_tests)
    _r(logistic_regression.run_tests)
    _r(svm.run_tests)

    # TODO: tests for lda, pca
