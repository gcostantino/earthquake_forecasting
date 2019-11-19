import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from data_preprocessing import trim_nans, standardize
from derivative import smoothing_derivative
from read_data import load

if __name__ == '__main__':
    X, y = load(os.path.join('..', 'datasets', 'dataset2_3033_20'))  # example dataset
    X, y = trim_nans(X, y)  # nan removal
    X = np.delete(X, range(10, 52), 1)
    #X = np.concatenate((X, np.linspace(0, X.shape[0], X.shape[0]).reshape(-1,1)), axis=1)
    # displacement rate as target
    for i in range(y.shape[1]):
        y[:, i] = smoothing_derivative(y[:, i], np.linspace(0, len(y[:, i]), len(y[:, i])), 30)

    frac_train = 0.6
    ind_train = int(frac_train * X.shape[0])
    Xtrain, ytrain, Xtest, ytest = X[:ind_train], y[:ind_train], X[ind_train:], y[ind_train:]
    Xtrain, _, Xtest, _ = standardize(Xtrain, Xtrain, Xtest)

    model = LinearRegression(fit_intercept=False)  # already centered
    model.fit(Xtrain, ytrain)  # multivariate
    # print("Coefficients:", model.coef_)
    ypred = model.predict(Xtest)
    plt.plot(ytest, label='True')
    plt.plot(ypred, label='Predicted')
    plt.legend()
    plt.show()

    r_squared = r2_score(ytest, ypred, multioutput='raw_values')
    print(type(r_squared))
    adj_r_squared = 1 - (1 - r_squared) * ((Xtest.shape[0] - 1) / (Xtest.shape[0] - Xtest.shape[1] - 1))
    print("R^2", r_squared)
    print("Adj R^2", adj_r_squared)
