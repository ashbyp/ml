import csv
from os import path
from urllib.request import urlopen

import numpy as np
import pandas as pd

FILENAME = 'spambase.data'


def download_spam(filename=FILENAME, add_header=False, force_download=False):
    if not path.exists(filename) or force_download:
        print(f'Downloading {FILENAME}')
        count = 0
        data = urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data')
        with open(filename, "wb") as f:
            if add_header:
                f.write(b'header\n')
                count += 1
            for line in data:
                count += 1
                f.write(line)
        return filename, count

    return filename, sum(1 for line in open(FILENAME))


def open_as_csv(filename=FILENAME):
    with open(filename, 'r') as f:
        data = list(csv.reader(f, delimiter=','))
        print(len(data))
    return np.array(data)


def split_features_and_labels(data):
    n_samples, n_features = data.shape
    n_features -= 1
    X = data[:,0:n_features]
    y = data[:, n_features]
    return X, y


def open_with_np(filename=FILENAME):
    return np.loadtxt(filename, delimiter=',')


def open_with_np_gen(filename=FILENAME, header=0):
    return np.genfromtxt(filename, delimiter=',', dtype=np.float32, skip_header=header)


def open_with_pandas(filename=FILENAME):
    df = pd.read_csv(filename, delimiter=',', header=None)
    return df.to_numpy()


def main():
    print(f'Downloaded: {download_spam(force_download=True)[1]} rows')

    data = open_as_csv()
    print(f'CSV data shape: {data.shape}')
    X, y = split_features_and_labels(data)
    print(f'X: {X.shape}')
    print(f'y: {y.shape}')

    data = open_with_np()
    print(f'NP data shape: {data.shape}')

    data = open_with_np_gen()
    print(f'NP Gen data shape: {data.shape} of type {type(data[0][0])}')

    data = open_with_pandas()
    print(f'Pandas data shape: {data.shape}')

    print(f'Downloaded: {download_spam(FILENAME + ".hdr", True)} rows')
    data = open_with_np_gen(filename=FILENAME + ".hdr", header=1)
    print(f'NP Gen data shape: {data.shape} of type {type(data[0][0])}')


if __name__ == '__main__':
    main()
