import csv
from os import path
from urllib.request import urlopen

import numpy as np
import pandas as pd

UCI_DATA = {
    'spam': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
        'header': False,
        'split_labels': True,
        'filename': 'spambase.data'
    },
    'tumor': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/primary-tumor/primary-tumor.data',
        'header': False,
        'split_labels': True,
        'filename': 'tumor.data'
    }
}


def download_url(url, filename, force_download):
    if not path.exists(filename) or force_download:
        print(f'Downloading {filename}')
        count = 0
        data = urlopen(url)
        with open(filename, "wb") as f:
            for line in data:
                count += 1
                f.write(line)
        return filename, count

    return filename, sum(1 for line in open(filename))


def load_uci(name, force_download=False):
    lines = download_url(UCI_DATA[name]['url'], UCI_DATA[name]['filename'], force_download)
    if not lines:
        raise RuntimeError(f'no data found for {name}')

    data = open_with_np(UCI_DATA[name]['filename'])

    if UCI_DATA[name]['split_labels']:
        return split_features_and_labels(data)

    return data


def open_as_csv(filename):
    with open(filename, 'r') as f:
        data = list(csv.reader(f, delimiter=','))
        print(len(data))
    return np.array(data)


def split_features_and_labels(data):
    n_samples, n_features = data.shape
    n_features -= 1
    X = data[:, 0:n_features]
    y = data[:, n_features]
    return X, y


def open_with_np(filename):
    return np.loadtxt(filename, delimiter=',')


def open_with_np_gen(filename, header=0):
    return np.genfromtxt(filename, delimiter=',', dtype=np.float32, skip_header=header)


def open_with_pandas(filename):
    df = pd.read_csv(filename, delimiter=',', header=None)
    return df.to_numpy()
