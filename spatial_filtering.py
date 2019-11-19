"""
Spatial dataset construction script, associated with the script "datasets.py". Each spatial dataset is built by
considering different spatial configurations, exported in the .pickle files in the directory ./datasets/spatial
whose names are linked to the final (features, target) dataset (ex. spatial1, etc.).
The Japan Meteorological Agency (JMA) catalog is used for extracting the data.
Given a certain station S, an earthquake is considered to affect the deformation registered at the station S over a
certain threshold deformation (supposed 1 mm) u_thresh if
    1/r^2 * 10^(3/2*m) > 10^3 * u_thresh,
where r is the distance between the earthquake hypocenter and the station and m its magnitude.
The task of spatial datasets is accomplished by using a common function (spatial_dataset) and a variable filtering
function (filtering_condition).
"""
import os
import scipy.io
from datetime import datetime, timedelta
import numpy as np
from read_data import save, load, haversine, read_stations_coordinates, read_stations
from map import show

# base_folder = os.path.join('.', 'datasets','spatial')
base_folder = os.path.join('datasets')

boso_box = [35, 35.7, 140, 140.7]
threshold = 1  # 1 mm
stations_list = ['3033', '3024', '3041', '0226']
station_coords = {}  # dict with mapping station - coordinates


def station_coordinates(station):
    """Returns latitude, longitude of the station passed as argument."""
    if station not in station_coords:  # read from file only the first time
        coordinates = read_stations_coordinates()
        stations = read_stations()
        coord_station = coordinates[stations.index(station)]
        p_station = coord_station[0:2]
        station_coords[station] = p_station

    return station_coords[station]


def centroid_stations(stations_list):
    """Computes the centroid of the positions of the stations passed as argument."""

    positions = []
    for station in stations_list:
        positions.append(station_coordinates(station))
    return np.mean(positions, axis=0)


centroid = centroid_stations(stations_list)


def _read_catalog():
    """Function which reads the JMA catalog (saved in a .mat file) at "./datasets/catalogs" and converts it into the
    pythonic-equivalent. Returns a tuple whose elements are lists containing the latitudes, longitudes, magnitudes Mw,
    depths and dates of occurrence, respectively."""
    path = os.path.join(base_folder, 'catalogs', 'jma_cat_1980_2013.mat')
    mat = scipy.io.loadmat(path)  # opened as a dict
    lat = mat['lat'].flatten()
    lon = mat['lon'].flatten()
    mw = mat['m'].flatten()
    depth = mat['z'].flatten()
    matlab_datenum = mat['t'].flatten()
    # matlab to python date conversion
    dates = [datetime.fromordinal(int(d)) + timedelta(days=d % 1) - timedelta(days=366) for d in matlab_datenum]
    dates = [d.replace(hour=0, minute=0, second=0, microsecond=0) for d in dates]
    return lat, lon, mw, depth, dates


def _cut_catalog():
    """Function which performs a cut on the original JMA catalog (for more efficient operations).
    The considered time period spans from 2000 (after the Izu volcanic activity) to 2010 (1 year before Tohoku disaster).
    The new catalog is saved as "2000-2010_catalog.pickle" in the directory "./datasets/catalogs"."""

    start_date = datetime.strptime('2000 9 10', '%Y %m %d')  # ~10 days margin from Izu volcanic activity end date
    end_date = datetime.strptime('2010 12 31', '%Y %m %d')
    lat, lon, mw, depth, dates = _read_catalog()

    curr_time = start_date
    lat_cut, lon_cut, mw_cut, depth_cut, dates_cut = [], [], [], [], []  # dates useful for cutting target series

    npdates = np.array(dates)  # conversion in ndarray for efficient search of occurrences of dates
    nplat, nplon, npmw, npdepth = np.array(lat), np.array(lon), np.array(mw), np.array(depth)

    while curr_time <= end_date:
        indices = np.where(npdates == curr_time)[0]
        lat_cut.append(nplat[indices].tolist())  # each day will correspond to a list
        lon_cut.append(nplon[indices].tolist())
        mw_cut.append(npmw[indices].tolist())
        depth_cut.append(npdepth[indices].tolist())
        dates_cut.append(curr_time)
        curr_time += timedelta(days=1)

    save_path = os.path.join(base_folder, 'catalogs', '2000-2010_catalog')
    save((lat_cut, lon_cut, mw_cut, depth_cut, dates_cut), save_path)


def _update_kwargs(**kwargs):
    """Updates kwargs dict with computed GPS distance, if needed."""
    if 'distance' not in kwargs:  # set distance the first time
        if kwargs['version'] == '0':
            kwargs['distance'] = compute_distance(station_coordinates(kwargs['station']),
                                                  (kwargs['lat'], kwargs['lon']))
        else:  # spatial 4 dataset
            kwargs['distance'] = compute_distance(kwargs['centroid'], (kwargs['lat'], kwargs['lon']))
    return kwargs


def compute_distance(p1, p2):
    """Computes gps distance between two points, expressed as (latitude, longitude), using haversine function."""

    return haversine(p1[1], p1[0], p2[1], p2[0])


def distance_magnitude_filtering(**kwargs):
    """Computes the spatial filtering taking into account the magnitude. Returns true if the point (lat, lon) is far
    from the station (lat_s, lon_s) at most radius and if 1/radius^2 * 10^(3/2*Mw) > 10^3 (threshold = 1 mm).
    Uses the haversine method for computing the GPS distance."""
    kwargs = _update_kwargs(**kwargs)
    return distance_condition(**kwargs) and 10 ** (1.5 * float(kwargs['magnitude'])) > 1000 * (
            float(kwargs['distance']) ** 2)


def boso_condition(**kwargs):
    """Aid for building "spatial1" dataset, which corresponds to "dataset1". It is built by considering all the
    earthquakes occurring in the Boso peninsula patch, i.e. a box having latitudes in the range [35 - 35.7] and
    longitudes in the range [140 - 140.7]."""

    return boso_box[0] < kwargs['lat'] < boso_box[1] and boso_box[2] < kwargs['lon'] < boso_box[3]


def distance_magnitude_condition(**kwargs):
    """Aid for building "spatial2" dataset, which corresponds to "dataset2". It is built by considering all the
    earthquakes at distance r (radius) from a given station and with magnitude m, such that 1/r^2 * 10^(3/2*m) > 10^3."""

    return distance_magnitude_filtering(**kwargs)


def distance_condition(**kwargs):
    """Returns true if the earthquake (lat, lon) is at distance at most radius from the station."""
    kwargs = _update_kwargs(**kwargs)
    return kwargs['distance'] < float(kwargs['radius'])


def joint_condition(**kwargs):
    """Aid for building "spatial4" dataset, which corresponds to "dataset4".
    Version 1: it is built by considering all the earthquakes at distance r (radius) from the centroid of the stations.
    Version 2: it is built by considering the earthquakes at distance r (radius) from the centroid of the stations which
    satisfy the distance condition, like spatial2 dataset."""

    if kwargs['version'] == 'v1':
        return distance_condition(**kwargs)
    elif kwargs['version'] == 'v2':
        return distance_magnitude_condition(**kwargs)
    else:
        return False


def filtering_condition(**kwargs):
    """Dispatching function for selecting the proper filtering strategy. Each function called receives two coordinates
    and returns a boolean, i.e. the condition for filtering the hypocenter identified by that coordinates."""

    options = {'1': boso_condition,
               '2_' + kwargs['station'] + '_' + kwargs['radius']: distance_magnitude_condition,
               '4_' + kwargs['version'] + '_' + kwargs['radius']: joint_condition,
               }
    return options[kwargs['filtering_type']](**kwargs)


def spatial_dataset(**kwargs):
    """Function used for building all the different spatial datasets. Uses a filtering function adapted for accomplishing
    the different filtering strategies.
    Filtering types: 1, 2_station_radius, 3_station, 4_version_radius."""

    # check for already set values and put neutral values
    if 'version' not in kwargs: kwargs['version'] = '0'
    if 'radius' not in kwargs: kwargs['radius'] = '0'
    if 'station' not in kwargs: kwargs['station'] = ''

    cut_catalog = os.path.join(base_folder, 'catalogs', '2000-2010_catalog')
    lat, lon, mw, depth, dates = load(cut_catalog)
    lat_filtered, lon_filtered, mw_filtered, depth_filtered, dates_filtered = [], [], [], [], []
    for i in range(len(dates)):  # each day corresponds to a list of events
        dates_filtered.append(dates[i])
        lat_filtered.append([])
        lon_filtered.append([])
        mw_filtered.append([])
        depth_filtered.append([])
        for j in range(len(lat[i])):  # for each event
            if filtering_condition(**kwargs, lat=lat[i][j], lon=lon[i][j], magnitude=mw[i][j], centroid=centroid):
                lat_filtered[i].append(lat[i][j])
                lon_filtered[i].append(lon[i][j])
                mw_filtered[i].append(mw[i][j])
                depth_filtered[i].append(depth[i][j])

    # show([j for i in lat_filtered for j in i], [j for i in lon_filtered for j in i],
    #      size=[j + 1 for i in mw_filtered for j in i])  # +1 added because min(mw)=-1
    save_path = os.path.join(base_folder, 'spatial', 'spatial' + kwargs['filtering_type'])
    # save((lat_filtered, lon_filtered, mw_filtered, depth_filtered, dates_filtered), save_path)


if __name__ == '__main__':

    spatial_dataset(filtering_type='1')
    radius_list = ['20', '30', '50', '100', '200']
    for station in stations_list:
        for radius in radius_list:
            spatial_dataset(filtering_type='2_' + station + '_' + radius, station=station, radius=radius)

    versions_list = ['v1', 'v2']
    radius_list = ['50', '100', '200']
    for version in versions_list:
        for radius in radius_list:
            spatial_dataset(filtering_type='4_' + version + '_' + radius, radius=radius, version=version)
