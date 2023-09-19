import pickle
import numpy as np


# Functions for reading cross-spectral analysis data saved onto 
# latitude-band tiles by csagan_multiprocess.py
# and cropping to specified lat/lon box.


def read_saved_data(save_filename):
    """
    Load pickle file in read mode.
    Parameters
    ----------
    save_filename: str
        Path to .p file to load
    Returns
    -------
    saved_data: unknown dtype
        Data loaded from pickle file
    """
    saved_data = pickle.load(open(save_filename, 'rb'))
    return saved_data


def crop_to_region(spectra_data, tile, lon_west, lon_east, lat_south, lat_north, resolution=0.25):
    """
    Crop data on one of the four latitude band tiles ('polar', 'northern', 'tropics', 'southern')
    to a sub-region of the tile.
    Doesn't test whether the region supplied is actually a subset of the tile.
    Parameters
    ----------
    spectra_data: numpy array (of any dtype)
        Data to crop (e.g. cross-spectral analysis results output).
        Area covered by data needs to match one of the tiles described below.
    tile: str
        Which tile the data is from.
        Tile naming scheme used for saving cross-spectal analysis output is:
        southern = 60S-25S
        tropics = 35S-35N
        northern = 25N-65S
        polar = 55N-80N
    lon_west: float
        Western longitude boundary of cropped region to return (degrees east, -180 to 180)
    lon_east: float
        Eastern longitude boundary of cropped region to return (degrees east, -180 to 180)
    lat_south: float
        Southern latitude boundary of cropped region to return (degrees north)
    lat_north: float
        Northern latitude boundary of cropped region to return (degrees north)
    resolution (kwarg, default=0.25): float
        Horizontal resolution of data being cropped
    Returns
    -------
    region_lats: numpy array (1D, float)
        Latitude coordinates of data in cropped region
    region_lons: numpy array (1D, float)
        Longitude coordinates of data in cropped region
    region_data: numpy array
        spectra_data cropped to specified region
    """
    if tile == 'tropics':
        tile_lat_south = -35
        tile_lat_north = 35
    elif tile == 'northern':
        tile_lat_south = 25
        tile_lat_north = 65
    elif tile == 'southern':
        tile_lat_south = -60
        tile_lat_north = -25
    elif tile == 'polar':
        tile_lat_south = 55
        tile_lat_north = 80
    tile_lats = np.arange(tile_lat_south, tile_lat_north, resolution) + 0.5*resolution
    tile_lons = np.arange(-180, 180, resolution) + 0.5*resolution
    lon_region_idcs = np.where(np.logical_and(tile_lons > lon_west, tile_lons < lon_east))[0]
    lat_region_idcs = np.where(np.logical_and(tile_lats > lat_south, tile_lats < lat_north))[0]
    lat_slice = slice(lat_region_idcs[0], lat_region_idcs[-1]+1)
    lon_slice = slice(lon_region_idcs[0], lon_region_idcs[-1]+1)
    region_lats = tile_lats[lat_slice]
    region_lons = tile_lons[lon_slice]
    region_data = spectra_data[lat_slice, lon_slice]
    return region_lats, region_lons, region_data


def read_region_data(save_filename, tile, lon_west, lon_east, lat_south, lat_north, resolution=0.25):
    """
    Read tile data from a pickle file and crop to a sub-region of the tile.
    Useful for cropping the full tiles, which overlap slightly for the purposes of 
    neighbourhood testing, to non-overlapping verions that can be concatenated into a global map.
    Parameters
    ----------
    save_filename: str
        Path to filename for tile data
    tile: str
        Which tile the data is from.
        Tile naming scheme used for saving cross-spectal analysis output is:
        southern = 60S-25S
        tropics = 35S-35N
        northern = 25N-65S
        polar = 55N-80N
    lon_west: float
        Western longitude boundary of cropped region to return (degrees east, -180 to 180)
    lon_east: float
        Eastern longitude boundary of cropped region to return (degrees east, -180 to 180)
    lat_south: float
        Southern latitude boundary of cropped region to return (degrees north)
    lat_north: float
        Northern latitude boundary of cropped region to return (degrees north)
    resolution (kwarg, default=0.25): float
        Horizontal resolution of data being cropped
    Returns
    -------
    region_lats: numpy array (1D, float)
        Latitude coordinates of data in cropped region
    region_lons: numpy array (1D, float)
        Longitude coordinates of data in cropped region
    region_data: numpy array
        spectra_data cropped to specified region
    """
    saved_data = read_saved_data(save_filename)
    region_lats, region_lons, region_data = crop_to_region(saved_data, tile, lon_west, lon_east, lat_south, lat_north,
                                                           resolution=resolution)
    return region_lats, region_lons, region_data
