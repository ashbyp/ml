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
        'filename': 'spambase.data',
        'missing': None,
        'filling': None,
        'labels_last': True
    },
    'wine': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
        'header': False,
        'split_labels': True,
        'filename': 'wine.data',
        'missing': None,
        'filling': None,
        'labels_last': False
    },
    'tumor': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/primary-tumor/primary-tumor.data',
        'header': False,
        'split_labels': True,
        'filename': 'tumor.data',
        'missing': ['?'],
        'filling': [1],
        'labels_last': True
    },
    'SPECT heart': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/spect',
        'files': ['SPECT.train', 'SPECT.test'],
        'header': False,
        'split_labels': True,
        'filename': 'spect_heart.data',
        'missing': None,
        'filling': None,
        'labels_last': False
    },
}


def download_urls(urls, filename, force_download):
    if not path.exists(filename) or force_download:
        print(f' --> Downloading {filename}')
        count = 0

        with open(filename, "wb") as f:
            for url in urls:
                data = urlopen(url)
                for line in data:
                    count += 1
                    f.write(line)
                f.write(b'\n')
        print(f' --> {count} samples downloaded')
        return filename, count

    return filename, sum(1 for line in open(filename))


def load_uci(name, force_download=False):
    data_def = UCI_DATA[name]
    if 'files' in data_def:
        urls = [data_def['url'] + '/' + file for file in data_def['files']]
    else:
        urls = [data_def['url']]

    _, lines = download_urls(urls, data_def['filename'], force_download)
    if not lines:
        raise RuntimeError(f'no data found for {name}')

    data = open_with_np_gen(data_def['filename'],
                            missing=data_def['missing'],
                            filling=data_def['filling'])

    if data_def['split_labels']:
        return split_features_and_labels(data, data_def['labels_last'])

    return data


def open_as_csv(filename):
    with open(filename, 'r') as f:
        data = list(csv.reader(f, delimiter=','))
        print(len(data))
    return np.array(data)


def split_features_and_labels(data, labels_last=True):
    n_samples, n_features = data.shape
    if labels_last:
        X = data[:, 0:n_features-1]
        y = data[:, n_features-1]
    else:
        X = data[:, 1:n_features]
        y = data[:, 0]

    return X, y


def open_with_np(filename):
    return np.loadtxt(filename, delimiter=',')


def open_with_np_gen(filename, header=0, missing=None, filling=None):
    return np.genfromtxt(filename, delimiter=',', dtype=np.float32, skip_header=header,
                         missing_values=missing, filling_values=filling)


def open_with_pandas(filename):
    df = pd.read_csv(filename, delimiter=',', header=None)
    return df.to_numpy()


if __name__ == '__main__':
    load_uci('spam', True)
    load_uci('SPECT heart', True)
