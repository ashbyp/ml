import urllib.request
from urllib.request import urlopen
import csv
import numpy as np
import pandas as pd

FILENAME = 'spambase.data'


def download_spam(filename=FILENAME, add_header=False):
    count = 0
    data = urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data')
    with open(filename, "wb") as f:
        if add_header:
            f.write(b'header\n')
            count += 1
        for line in data:
            count += 1
            f.write(line)
    return count


def open_as_csv():
    with open(FILENAME, 'r') as f:
        data = list(csv.reader(f, delimiter=','))
        print(len(data))
    return np.array(data)


def split_samples_and_features(data):
    n_samples, n_features = data.shape
    n_features -= 1
    X = data[:,0:n_features]
    y = data[:, n_features]
    return X, y


def open_with_np():
    return np.loadtxt(FILENAME, delimiter=',')


def open_with_np_gen(filename=FILENAME, header=0):
    return np.genfromtxt(filename, delimiter=',', dtype=np.float32, skip_header=header)


def open_with_pandas():
    df = pd.read_csv(FILENAME, delimiter=',', header=None)
    return df.to_numpy()


def main():
    print(f'Downloaded: {download_spam()} rows')

    data = open_as_csv()
    print(f'CSV data shape: {data.shape}')
    X, y = split_samples_and_features(data)
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
