import numpy as np


def movmean(x, k):
    """Effettua la media mobile centrata di lunghezza k (dispari)"""
    if k % 2 == 0:
        raise Exception('k non dispari')
    x_m = []
    amp = (k - 1) // 2
    for i in range(len(x)):
        # controllo sugli estremi
        if i - amp < 0:
            x_m.append(np.nanmean(x[0:i + amp]))
        elif i + amp > len(x):
            x_m.append(np.nanmean(x[i - amp:]))
        else:
            x_m.append(np.nanmean(x[i - amp:i + amp]))
    return np.array(x_m)


'''from read_data import read_time_series_station
import matplotlib.pyplot as p

a, c = read_time_series_station('0226')
x = a[:, 1]
m=movmean(x, 5001)
p.subplot(2, 1, 1)
p.plot(x)
p.subplot(2, 1, 2)
p.plot(m)
p.show()
print(m[473-1:482-1])'''