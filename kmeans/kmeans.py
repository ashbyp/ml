from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from util.common import class_name, accuracy

from util.common import euclidean_distance
from util.data import load_uci

np.random.seed(42)


class NumpyKMeans:

    def __init__(self, K=5, max_iterations=100, plot_steps=False):
        self.K = K
        self.max_iterations = max_iterations
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        self.X = None
        self.n_samples, self.n_features = None, None

    def fit_predict(self, X):
        self.X = X
        self.n_samples, self.n_features = self.X.shape

        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[i] for i in random_sample_idxs]

        for _ in range(self.max_iterations):
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self.plot_steps:
                self.plot()
            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids_new):
        distances = [euclidean_distance(centroids_old[i], centroids_new[i]) for i in range(self.K)]
        return sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        plt.show()


def test(km, dataset_name, X, y, verbose=False):
    labels = km.fit_predict(X)

    if verbose:
        print(f'Labels: {labels}')
        if hasattr(km, 'plot'):
            km.plot()

    print(f'{class_name(km)} labels counter for {dataset_name} {sorted(Counter(labels).values())}')


def run_tests(verbose=False):
    X, y = datasets.make_blobs(n_samples=200, n_features=3, centers=8, shuffle=True, random_state=42)
    clusters = len(np.unique(y))

    if verbose:
        print(f'X_shape:  {X.shape}')
        print(f'Clusters: {clusters}')

    test(NumpyKMeans(K=clusters, max_iterations=150, plot_steps=False), "blobs", X, y, verbose)
    test(KMeans(n_clusters=clusters, random_state=0), "blobs", X, y, verbose)


if __name__ == '__main__':
    run_tests(verbose=True)
