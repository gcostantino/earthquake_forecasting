"""
Dataset construction script. Each dataset in the form (features, target) is contained in a .pickle file in the directory
./datasets/ and referenced by a number and optional parameters (ex. dataset1 or dataset2_3033_200). Such dataset is
built by considering different spatial configurations, contained in the .pickle files in the directory ./datasets/spatial
whose names are linked to the dataset (ex. spatial1, etc.) and build using the script "spatial_filtering.py".
The Japan Meteorological Agency (JMA) catalog is used for extracting the data.
"""
import os

import numpy as np

from derivative import smoothing_derivative
from period import eqs_day_range
from read_data import load, save, read_time_series_station, read_time, day_time_dict

base_folder = os.path.join('datasets')
stations_list = ['3033', '3024', '3041', '0226']


def _Mw_to_Mo_list(l):
    """Converts a Mw_list in a Mo_list according to the equation Mo = 10^(3/2*(Mw + 6.07))"""
    new_list = []
    for elem in l:
        new_list.append(10 ** (3 / 2 * (elem + 6.07)))
    return new_list


def _station_target_extraction(**kwargs):
    """Extract target time series belonging to the station considered."""
    spatial_origin = kwargs['spatial_origin']  # spatial 2 used by default if dataset3 (radius 20, but it's equivalent)
    load_path = os.path.join(base_folder, 'spatial',
                             spatial_origin if spatial_origin != '' else 'spatial2_' + kwargs['station'] + '_20')
    dates = load(load_path)[-1]

    raw, _ = read_time_series_station(kwargs['station'])
    components = raw[:, 1:4].tolist()  # north component by default
    t = read_time()
    d = day_time_dict(t)  # conversion date-integer of time series temporal evolution

    y = []  # will contain 3 columns for the north, east, vertical components
    for i in range(len(components[0])):  # 3 components
        y.append([])
        for date in dates:
            y[i].append(components[t.index(d[date])][i])

    y = np.array(y).T

    return y


def target_extraction(**kwargs):
    """Function which extracts the target time series according to the type of dataset considered."""

    if kwargs['base'] != 'spatial4':
        y = _station_target_extraction(**kwargs)
    else:
        y = _station_target_extraction(**kwargs, station=stations_list[0])
        for station in stations_list[1:]:
            y_station = _station_target_extraction(**kwargs, station=station)
            y = np.concatenate((y, y_station), axis=1)

    return y


def feature_extraction(**kwargs):
    """Generic function for features extraction from a spatial dataset defined by the script "spatial_filtering.py".
        The feature extracted are:

            - #eqs
            - cum #eqs
            - #eqs in a time window (tw= 10 days, 30 days, 60 days, 180days, 1 year, 2 years, 3 years, 4 years)
            - eq rate in different tw
            - eq acceleration in different tw
            - sum Mo per day
            - cum sum Mo per day
            - sum Mo in different tw
            - Mo rate in different tw
            - Mo acceleration in different tw

        for a total of 52 features."""

    features = {}  # organized as a dict for better re-usage
    tw = [10, 30, 60, 180, 365, 365 * 2, 365 * 3, 365 * 4]

    spatial_dataset = kwargs['spatial_origin']
    load_path = os.path.join(base_folder, 'spatial', spatial_dataset)

    lat, lon, mw, depth, dates = load(load_path)

    time_array = np.linspace(0, len(dates),
                             len(dates))  # numerical array for linregress applied in smoothing_derivative

    n_eqs = [len(l) for l in lat]
    cum_eqs = np.cumsum(n_eqs).tolist()
    Mo_count = [np.sum(_Mw_to_Mo_list(l)) for l in mw]

    features['n_eqs'] = [n_eqs]
    features['cum_eq'] = [cum_eqs]
    features['eq_tw'] = eqs_day_range(features['n_eqs'][0], dates, tw)
    features['eq_rate'] = [smoothing_derivative(features['n_eqs'][0], time_array, win) for win in tw]
    features['eq_acc'] = [smoothing_derivative(rate, time_array, win) for (rate, win) in zip(features['eq_rate'], tw)]
    features['Mo_count'] = [Mo_count]
    features['Mo_cum'] = [np.cumsum(features['Mo_count'][0]).tolist()]
    features['Mo_tw'] = eqs_day_range(features['Mo_count'][0], dates, tw)  # reused eqs_range for Mo
    features['Mo_count_rate'] = [smoothing_derivative(features['Mo_count'][0], time_array, win) for win in tw]
    features['Mo_count_acc'] = [smoothing_derivative(rate, time_array, win) for (rate, win) in
                                zip(features['Mo_count_rate'], tw)]

    X = []
    for l in features.values():
        X += l
    X = np.array(X).T

    return X


def dataset_creation(**kwargs):
    """Generic function for dataset creation."""

    spatial_dataset = str(kwargs['spatial_origin'])
    base = spatial_dataset.split('_')[0]
    if base != '':
        X = feature_extraction(**kwargs)
    else:  # dataset 3
        radius_list = ['20', '30', '50', '100', '200']
        load_path = os.path.join(base_folder, 'dataset2_' + kwargs['station'] + '_' + radius_list[0])
        X, _ = load(load_path)
        for radius in radius_list[1:]:
            load_path = os.path.join(base_folder, 'dataset2_' + kwargs['station'] + '_' + radius)
            X_loaded, _ = load(load_path)
            X = np.concatenate((X, X_loaded), axis=1)
        spatial_dataset = 'dataset3_' + kwargs['station']

    # target
    y = target_extraction(**kwargs, base=base)

    dataset = (X, y)  # nan values not handled here

    save_path = os.path.join(base_folder, spatial_dataset.replace('spatial', 'dataset'))
    save(dataset, save_path)


if __name__ == '__main__':

    dataset_creation(spatial_origin='spatial1', station='3033')  # dataset 1
    radius_list = ['20', '30', '50', '100', '200']
    for station in stations_list:
        for radius in radius_list:
            dataset_creation(spatial_origin='spatial2_' + station + '_' + radius, station=station)  # dataset 2

    for station in stations_list:
        dataset_creation(spatial_origin='', station=station)  # dataset 3

    versions_list = ['v1', 'v2']
    radius_list = ['50', '100', '200']
    for version in versions_list:
        for radius in radius_list:
            dataset_creation(spatial_origin='spatial4_' + version + '_' + radius)  # dataset 2
