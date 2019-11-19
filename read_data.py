import numpy as np
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
import pickle as pk


def save(obj, name):
    with open(name + '.pickle', 'wb') as handle:
        pk.dump(obj, handle, protocol=pk.HIGHEST_PROTOCOL)


def load(name):
    with open(name + '.pickle', 'rb') as handle:
        obj = pk.load(handle)
    return obj


def _get_catalog():
    data = []
    full_data = []
    with open('cataloghi_sismici/eqdata', 'r') as f:
        for line in f:
            row = line.rstrip('\n').split('\t')
            row[0] = int(row[0])  # anno
            row[1] = int(row[1])  # mese
            row[2] = int(row[2])  # giorno
            row[3] = int(row[3])  # ora
            row[4] = int(row[4])  # minuto
            row[5] = float(row[5])  # secondo
            row[6] = float(row[6])  # magnitudo
            row[7] = float(row[7])  # latitudine
            row[8] = float(row[8])  # longitudine
            row[9] = float(row[9])  # profondita'
            full_data.append(row)
            new_row = [row[0], row[1], row[2], row[7], row[8], row[9], row[6]]
            data.append(new_row)
            # rimuovo data, ora e minuto per rendere uniforme con dati di Anne e rendo in formato:
            # anno, mese, giorno, latitudine, longitudine, profondita', magnitudo come dati di Anne
    return full_data, data


def read_full_catalog():
    return _get_catalog()[0]


def read_catalog():
    return _get_catalog()[1]


def read_time_series_all(station):
    series = np.loadtxt('dati_sismici/ALL_GEOPHY/' + station + '.txt')
    return series


def read_time_series_slow_def(station):
    series = np.loadtxt('dati_sismici/SLOW_DEF/' + station + '.txt')
    return series


def read_time_series_station(station):
    s_all = read_time_series_all(station)
    s_sd = read_time_series_slow_def(station)
    return s_all, s_sd

def directions():
    directions = {'N-S': 1, 'E-W': 2, 'U-D': 3}
    return directions


def read_time():
    t = read_time_series_all('0042')[:, 0]  # 0042 stazione a caso
    return t.tolist()


def read_stations():
    stations = np.loadtxt('dati_sismici/INPUT_FILES/stations_japan_all.txt', dtype=np.str).tolist()
    return stations


def read_stations_coordinates():
    coordinates = np.loadtxt('dati_sismici/INPUT_FILES/japan_all_coordinates.txt').tolist()
    return coordinates


def _read_sses():
    sse = []
    with open('dati_sismici/INPUT_FILES/sse.txt', 'r') as f:
        for line in f:
            row = line.rstrip('\n').split(' ')
            row[0] = int(row[0])  # anno
            row[1] = int(row[1])  # mese
            row[2] = int(row[2])  # giorno
            row[3] = float(row[3])  # latitudine
            row[4] = float(row[4])  # longitudine
            row[5] = float(row[5])  # magnitudo
            sse.append(row)
    return sse


def read_sses():
    sses = _read_sses()
    sses_dates = []
    d = day_time_dict(read_time())
    for row in sses:
        s = str(row[0]) + " " + str(row[1]) + " " + str(row[2])
        date = datetime.strptime(s, '%Y %m %d')
        sses_dates.append(d[date])
    return sses_dates


def get_full_info():
    data = read_full_catalog()
    dates = []
    depths = []
    magnitudes = []
    latitudes = []
    longitudes = []
    first_day = None
    last_day = None
    for i in range(len(data)):  # calcolo delle date per le ascisse (year - month - day - hour - minute - second)
        row = data[i]
        Y = str(row[0])
        m = '0' + str(row[1]) if row[1] < 10 else str(row[1])
        d = '0' + str(row[2]) if row[2] < 10 else str(row[2])
        H = '0' + str(row[3]) if row[3] < 10 else str(row[3])
        M = '0' + str(row[4]) if row[4] < 10 else str(row[4])
        S = '0' + str(int(row[5])) if row[5] < 10.0 else str(int(row[5]))  # considero solo la parte intera
        s = Y + " " + m + " " + d + ", " + H + ":" + M + ":" + S
        date_strp = datetime.strptime(s, '%Y %m %d, %H:%M:%S')
        dates.append(date_strp)
        depths.append(-row[9])  # considero segno negativo
        magnitudes.append(row[6])
        latitudes.append(row[7])
        longitudes.append(row[8])
        if i == 0:
            first_day = date_strp.replace(hour=0, minute=0, second=0)  # metto orario a 00:00:00 per creare fascia
        if i == len(data) - 1:
            last_day = date_strp.replace(hour=0, minute=0, second=0)  # metto orario a 00:00:00 per creare fascia
    return dates, depths, magnitudes, latitudes, longitudes, first_day, last_day


def eq_count_general():
    info = get_full_info()
    dates, first_day, last_day = info[0], info[-2], info[-1]
    days = []  # ogni elemento sara' il giorno, scorrendo tutto il catalogo
    dict_time = {i: tuple() for i in
                 range((last_day - first_day).days)}  # tengo traccia di un dizionario in cui la chiave sia una tupla
    # contenente la fascia oraria (da riempire successivamente e il valore sia l'indice della lista hours corrispondente
    # a quella fascia oraria. Per facilita' di creazione creiamo il dizionario inverso e poi lo invertiamo
    time = first_day
    i = 0
    while time < last_day:
        newtime = time + timedelta(days=1)
        dict_time[i] = (time, newtime)  # uso una tupla con gli estremi
        days.append(time)
        time = newtime
        i += 1

    dict_time_inv = {v: k for k, v in dict_time.items()}  # inverto il dizionario in quanto mi interessa la fascia
    ev_count = [0] * len(days)  # ogni elemento sara' il numero di eventi corrispondenti alla fascia oraria associata
    # all'indice
    for date in dates:  # scorro direttamente le date gia' create
        # creo fascia giornaliera
        lower = date.replace(hour=0, minute=0, second=0)  # metto orario a 00:00:00 per creare fascia
        upper = lower + timedelta(days=1)
        # aggiorno il conteggio
        if (lower, upper) in dict_time_inv:  # controllo per ultimo elemento
            ev_count[dict_time_inv[(lower, upper)]] += 1
    return ev_count


def day_time_dict(t):
    day_time = {}
    start_time = datetime.strptime('1995 1 1', '%Y %m %d')
    time = start_time
    for i in range(len(t)):
        day_time[time] = t[i]
        time += timedelta(days=1)
    return day_time


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) NON STANDARD
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km


def selection_criterion(eq, station, min_mw, eq_count_all, Mo_count_all, depth_list, Mo_list, date, coordinates, stations, day_time, t):
    # cerco tutte gli eventi che hanno tale data
    # tra tali eventi, seleziono solo quelli che hanno interessato la stazione in esame

    #################################################################################################################
    ############################################## CRITERIO DI SCELTA ###############################################
    # Se l'EQ e' large (Mw > 6.4 ad esempio), esso influisce sulla serie GPS se vale la seguente relazione empirica:
    # d(Mw) < 10^(Mw/2-0.8)
    # dove d(Mw) e' la distanza tra la stazione e l'EQ considerato (in km)
    # Mw e' la magnitudo dell'EQ considerato
    # Per EQ minori, invece, (Mw < 6.4), e' piu' tricky.
    # Se ipotizziamo un legame tra lo slow slip sulla subduzione e il tasso di sismicita', allora
    # dovremmo usare un modello a soglia e dunque considerare una distanza minima al di sotto della quale
    # Anne suggerisce di considerare tutti gli EQ.
    # Questa distanza potrebbe essere ad esempio 200km, ma possiamo giocare con questo valore. --> ML can help
    #################################################################################################################
    radius_threshold_smaller_eq = 20000  # km
    Mw_threshold = 6.4

    coord_station = coordinates[stations.index(station)]
    p_station = coord_station[0:2]
    H_station = coord_station[2]
    p_eq = eq[3:5]
    depth_eq = eq[5]
    Mw = int(eq[6])
    radius_threshold_large_eq = 10 ** (Mw / 2 - 0.8)  # km
    d_gps = haversine(p_station[1], p_station[0], p_eq[1], p_eq[0])  # non standard. Va cambiato
    d_depth = H_station + depth_eq

    #dist = max(d_gps, d_depth)
    dist = d_gps
    #print("scelto max tra dgps e ddepth", d_gps, d_depth)

    if Mw < min_mw:
        raise Exception("FATAL")
        #return

    Mo = 10 ** (3 / 2 * (Mw + 6.07))
    if Mw > Mw_threshold:  # large EQ
        #if d_gps < radius_threshold_large_eq and d_depth < radius_threshold_large_eq:  # Criterio
        if d_gps < radius_threshold_large_eq:
            eq_count_all[station][t.index(day_time[date])] += 1
            Mo_count_all[station][t.index(day_time[date])] += Mo
            depth_list[station][t.index(day_time[date])].append(depth_eq)
            Mo_list[station][t.index(day_time[date])].append(Mo)
    else:  # smaller EQ
        #if d_gps < radius_threshold_smaller_eq and d_depth < radius_threshold_smaller_eq:  # Criterio
        #print("dist<r:", dist, radius_threshold_smaller_eq)
        if dist < radius_threshold_smaller_eq:
            eq_count_all[station][t.index(day_time[date])] += 1
            Mo_count_all[station][t.index(day_time[date])] += Mo
            depth_list[station][t.index(day_time[date])].append(depth_eq)
            Mo_list[station][t.index(day_time[date])].append(Mo)





def Mo_to_cum_Mw(Mo_sum):
    # Mw_cum = 2/3 * log10(Mo_sum) - 6.07
    return ((2/3) * np.log10(Mo_sum)) - 6.07
    #return Mo_sum


def eq_count(station, min_mw = 0): # metto min 0 ma dovrebbe essere la mw di completezza
    data = read_catalog()
    stations = read_stations()
    t = read_time()
    coordinates = read_stations_coordinates()
    day_time = day_time_dict(t)

    eq_count_all = {s: [0] * len(t) for s in stations}  # dizionario dove per ogni stazione e' presente una lista dove
    Mo_count_all = {s: [0] * len(t) for s in stations} # per momento (somma momenti al giorno)
    depth_list = {s: [[] for _ in t] for s in stations}
    Mo_list = {s: [[] for _ in t] for s in stations}



    # FOR FUTURE DEVELOPMENT --> estendere eq_count_all

    # ogni indice e' il tempo, data l'associazione fatta in precedenza, e il valore e' il numero di eq
    # creo un vettore di n_eq statico, che sara' indicizzato grazie al dict creato prima
    for eq in data:
        date = datetime.strptime(str(eq[0]) + ' ' + str(eq[1]) + ' ' + str(eq[2]), '%Y %m %d')
        # la procedura selection_criterion e' tale per cui, a partire dal dizionario eq_count_all, lo modifica
        # in base al criterio di scelta
        selection_criterion(eq, station, min_mw, eq_count_all, Mo_count_all, depth_list, Mo_list, date, coordinates, stations, day_time, t)
    return eq_count_all[station], Mo_to_cum_Mw(np.cumsum(Mo_count_all[station])), Mo_count_all[station], depth_list[station], Mo_list[station]


def read_gardonio():
    """Restituisce un dizionario indicizzato per stazione contenente come valore [t,N-S,E-W,U-D]"""
    stations = []
    import os
    for file in os.listdir("pos_gardonio"):
        if file.endswith(".txt"):
            station = file.title().split('.')[0]
            stations.append(station)
    result = {}
    for station in stations:
        t = []
        ns = []
        ew = []
        ud = []
        with open('pos_gardonio/' + station + '.txt', 'r') as f:
            rows = f.readlines()[36:]
            for row in rows:
                r = row.split()  # numero arbitrario di spazi
                t.append(float(r[0]))
                ns.append(float(r[15]) * 1000)
                ew.append(float(r[16]) * 1000)
                ud.append(float(r[17]) * 1000)
        data = np.stack((t, ns, ew, ud), axis=1)
        result[station] = data
    return result

'''a,c=read_time_series_station('0226')
import matplotlib.pyplot as plt
plt.plot(read_time(),a[:,2], linewidth=0.4)
plt.show()'''