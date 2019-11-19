import numpy as np
from scipy.stats import stats

from detrend import detrend


def calculate_velocity_linreg2(time1, d, tsw):
    vnew, vnew2 = [], []
    aux = tsw // 2
    aux2 = 0
    for j in range(len(d)):
        if j >= 0 and j <= aux - 1:
            at = time1[j - aux2:j + aux]
            at = np.array(at)
            ad = d[j - aux2:j + aux]
            mask = ~np.isnan(at) & ~np.isnan(ad)
            if len(ad[mask]) == 0 or len(at[mask]) == 0:
                vnew.append(np.nan)
                aux2 = aux2 + 1
            else:
                slope, intercept, rvalue, pvalue, std_err = stats.linregress(at[mask], ad[mask])
                vnew.append(slope)
                aux2 = aux2 + 1
        elif j > len(d) - 1 - aux and j <= len(d) - 1:
            at2 = time1[j - aux:len(d) - 1]
            at2 = np.array(at2)
            ad2 = d[j - aux:len(d) - 1]
            mask = ~np.isnan(at2) & ~np.isnan(ad2)
            if len(ad2[mask]) == 0 or len(at2[mask]) == 0:
                vnew.append(np.nan)
            else:
                slope, intercept, rvalue, pvalue, std_err = stats.linregress(at2[mask], ad2[mask])
                vnew.append(slope)
        else:
            at3 = time1[j - aux:j + aux]
            at3 = np.array(at3)
            ad3 = d[j - aux:j + aux]
            mask = ~np.isnan(at3) & ~np.isnan(ad3)
            if len(ad3[mask]) == 0 or len(at3[mask]) == 0:
                vnew.append(np.nan)
            else:
                slope, intercept, rvalue, pvalue, std_err = stats.linregress(at3[mask], ad3[mask])
                vnew.append(slope)
    for jj in range(len(d)):
        if (np.isnan(d[jj])):
            vnew2.append(np.nan)
        else:
            vnew2.append(vnew[jj])
    return vnew2


def calculate_vel_movil(vn, ve, vv, tm):
    # tm must be par
    vn2 = []
    ve2 = []
    vv2 = []
    aux = tm // 2
    aux2 = 1
    for j in range(len(vn)):
        # Primer dato
        if j == 0:
            vn2.append(np.nanmean(vn[j:j + aux]))
        # Entre el primer dato y el tamaño de la media movil
        elif j >= 1 and j <= aux - 1:
            vn2.append(np.nanmean(vn[j - aux2:j + aux]))
            aux2 = aux2 + 1
        # Parte final de los datos
        elif j > len(vn) - 1 - aux and j < len(vn) - 1:
            vn2.append(np.nanmean(vn[j - aux:len(vn) - 1]))
        # Media movil parte final de la muestra
        elif j == len(vn) - 1:
            vn2.append(np.nanmean(vn[j - aux:j]))
        # Media movil normal
        else:
            vn2.append(np.nanmean(vn[j - aux:j + aux]))
    return vn2, ve2, vv2


def smoothing_derivative(x, t, T):
    der = calculate_velocity_linreg2(t, detrend(t, x), T)
    der_smooth, _, _ = calculate_vel_movil(der, der, der, T)
    return der_smooth
