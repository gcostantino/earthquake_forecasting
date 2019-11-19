from scipy import stats
import numpy as np

# https://stackoverflow.com/questions/44779315/detrending-data-with-nan-value-in-scipy-signal


def detrend(t, x):
    """Detrend di x(t)"""
    if type(t) != np.ndarray or type(x) != np.ndarray:
        t = np.array(t)
        x = np.array(x)

    not_nan_ind = ~np.isnan(x)
    m, b, r_val, p_val, std_err = stats.linregress(t[not_nan_ind], x[not_nan_ind])
    detrend_x = x - (m * t + b)
    return detrend_x