import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVR, LinearSVR

from data_preprocessing import trim_nans, standardize
from derivative import smoothing_derivative
from read_data import load

import random

def optimizer(Xtr, Xval, ytr, yval, **params):
    """Returns the model optimized with Random Mutation Hill Climber algorithm."""

    n_iter = 50000
    C = 680.7245797295011  # initial guess
    epsilon = 0.09096251676474679  # initial guess
    sigma_C = 5
    sigma_epsilon = 0.01

    best_C = C
    best_epsilon = epsilon

    model = LinearSVR(**params, C=C, epsilon=epsilon)
    model.fit(Xtr, ytr)
    score = model.score(Xval, yval)

    best_score = score
    best_model = model

    for i in range(n_iter):
        print("Iterazione", i) if i % 10000 == 0 else None
        r = random.random()  # with uniform probability C or epsilon will be changet at each iteration
        if r < 0.5:
            C = np.random.normal(C, sigma_C)  # random probability centered in old C with fixed std dev
            while C < 0:
                C = np.random.normal(C, sigma_C)
        else:
            epsilon = np.random.normal(epsilon, sigma_epsilon)
            while epsilon < 0 or epsilon > 0.01:
                epsilon = np.random.normal(epsilon, sigma_epsilon)

    model = LinearSVR(**params, C=C, epsilon=epsilon)
    model.fit(Xtr, ytr)
    score = model.score(Xval, yval)
    if score > best_score:
        best_score = score
        best_model = model
        C = C  # hill climber
        epsilon = epsilon  # hill climber
    print("Best parameters:", C, epsilon)
    return best_model


if __name__ == '__main__':
    X, y = load(os.path.join('..', 'datasets', 'dataset2_3033_20'))  # example dataset
    X, y = trim_nans(X, y)  # nan removal
    # X = np.delete(X, range(10, 52), 1)
    # displacement rate as target
    for i in range(y.shape[1]):
        y[:, i] = smoothing_derivative(y[:, i], np.linspace(0, len(y[:, i]), len(y[:, i])), 30)

    y = y[:, 0]  # only north component. SVR does not support multivariate prediction

    frac_train = 0.45
    frac_val = 0.20
    ind_train = int(frac_train * X.shape[0])
    ind_val = int(frac_val * X.shape[0]) + ind_train
    Xtrain, ytrain, Xval, yval, Xtest, ytest = X[:ind_train], y[:ind_train], X[ind_train:ind_val], y[ind_train:ind_val], X[ind_val:], y[ind_val:]
    Xtrain, Xval, Xtest, _ = standardize(Xtrain, Xval, Xtest)

    model = optimizer(Xtrain, Xval, ytrain, yval, loss='epsilon_insensitive', random_state=42, dual=True, fit_intercept=True)


    ypred = model.predict(Xtest)
    plt.plot(ytest, label='True')
    plt.plot(ypred, label='Predicted')
    plt.legend()
    plt.show()

    ypred_val = model.predict(Xval)
    plt.plot(yval, label='True')
    plt.plot(ypred_val, label='Predicted')
    plt.legend()
    plt.title("Performance on validation set")
    plt.show()

    r_squared = r2_score(ytest, ypred, multioutput='raw_values')
    print(type(r_squared))
    adj_r_squared = 1 - (1 - r_squared) * ((Xtest.shape[0] - 1) / (Xtest.shape[0] - Xtest.shape[1] - 1))
    print("R^2", r_squared)
    print("Adj R^2", adj_r_squared)
