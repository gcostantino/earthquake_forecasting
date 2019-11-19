import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from read_data import load

import itertools


def trim_nans(X, y):
    """Removes nan values from the dataset (X, y). The target series y can have missing values, therefore for each column
    of the target series, the rows corrisponding to nan values are trimmed."""

    ind = np.zeros((y.shape[0],), dtype=bool)  # for keeping track of non-nans rows
    for i in range(y.shape[1]):
        ind = ind | ~np.isnan(y[:, i])
    return X[ind], y[ind]


def standardize(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler


if __name__ == '__main__':
    X, y = load(os.path.join('datasets', 'dataset2_3033_20'))  # example dataset

    # nan removal

    X, y = trim_nans(X, y)

    # plt.plot(X[:,3])
    # plt.show()

    #X = X[:, 40:46]
    print(X.shape)

    X = np.delete(X, range(28, 52), 1)

    print(X.shape)

    Rx = np.corrcoef(X, rowvar=False)
    plt.imshow(Rx)
    plt.colorbar()
    plt.show()
    data = pd.DataFrame(X)
    #sns.pairplot(data)
    #plt.show()
    import time
    from scipy.stats import pearsonr as p
    '''for comb in itertools.combinations([X[:,i] for i in range(X.shape[1])], 2):
        plt.scatter(comb[0], comb[1])
        plt.show()
        time.sleep(3)'''
    plt.scatter(X[:, 2], X[:, 28])
    print(p(X[:, 2], X[:, 28]))
    plt.show()
