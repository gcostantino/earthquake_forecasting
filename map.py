import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def show(lat, lon, size=2):
    """Overlay points with latitudes and longitudes given by lat, lon onto the Boso map."""

    #bounding_box = (138.5, 141.5, 34.2, 36.8)  # min lon, max lon, min lat, max lat
    bounding_box = (138, 143, 33.5, 37.5)  # min lon, max lon, min lat, max lat
    boso = plt.imread('boso_map2.png')

    fig, ax = plt.subplots()
    ax.set_title('Earthquakes in Boso peninsula')
    ax.set_xlim(bounding_box[0], bounding_box[1])
    ax.set_ylim(bounding_box[2], bounding_box[3])
    ax.scatter(lon, lat, zorder=1, alpha=0.2, c='r', s=size, edgecolor='r')
    ax.imshow(boso, zorder=0, extent=bounding_box, aspect='equal')
    plt.show()

if __name__ == '__main__':
    lats = np.random.sample((50,)) * (36.8 - 34.2) + 34.2
    lons = np.random.sample((50,)) * (141.5 - 138.5) + 138.5
    show(lats, lons)
