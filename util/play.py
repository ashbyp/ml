from sklearn import datasets


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # print(X)
    # print(y)

    print(X[0])
    print(X[1])

    print()


    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

