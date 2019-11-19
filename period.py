import numpy as np

def period_center_matrix(t, n_days, n_eq):
    m = np.zeros((n_days, len(t)))
    for j in range(len(t)):
        # creo una finestra temporale che inizia n_days/2 giorni prima del giorno in questione e termina n_days/2 giorni dopo
        # in m[i,j] ci sara' la somma cumulativa parziale partendo dal giorno j con durata i giorni
        # Vale la relazione m[i,j] = m[i-1,j] + n_eq[j], dove n_eq[j] e' il numero di eq al giorno j
        for i in range(n_days):
            if i == 0:
                m[i, j] = n_eq[j]
            else:
                if i % 2 != 0:  # dispari
                    m[i, j] = m[i - 1, j]
                    if j + i // 2 < len(t):
                        m[i, j] += n_eq[j + i // 2]
                else:  # pari
                    if i >= 4:  # aggiungo solo gli estremi
                        m[i, j] = m[i - 2, j]
                        if j + i // 2 < len(t):
                            m[i,j] += n_eq[j + i // 2]
                        if j - i // 2 > 0:
                            m[i, j] += n_eq[j - i // 2]
                    else:  # i=2
                        if j + 1 < len(t):
                            m[i, j] += n_eq[j + 1]
                        if j - 1 > 0:
                            m[i, j] += n_eq[j - 1]
    return m

def period_right_matrix(t, n_days, n_eq):
    m = np.zeros((n_days, len(t)))
    for j in range(len(t)):
        # creo una finestra temporale che inizia il giorno in questione e termina al piu' due anni dopo
        # in m[i,j] ci sara' la somma cumulativa parziale partendo dal giorno j con durata i giorni
        # Vale la relazione m[i,j] = m[i-1,j] + n_eq[j+i], dove n_eq[j+i] e' il numero di eq al giorno j+i
        # if j % 200 == 0:
        # print("Analizzato tempo t:", j)
        for i in range(n_days):
            if i + j < len(t):
                m[i, j] = m[i - 1, j] + n_eq[j + i]
    return m

def eqs_day_range(n_eq, t, tw):
    # computed complete matrix and extracted only rows of interest
    m = period_center_matrix(t, tw[-1], n_eq)
    res = []
    for win in tw:
        res.append(m[win - 1, :])
    return res
