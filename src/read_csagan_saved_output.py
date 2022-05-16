import pickle
import numpy as np


def read_saved_data(save_filename):
    saved_data = pickle.load(open(save_filename, 'rb'))
    return saved_data


def crop_to_region(spectra_data, tile, lon_west, lon_east, lat_south, lat_north, resolution=0.25):
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
    saved_data = read_saved_data(save_filename)
    region_lats, region_lons, region_data = crop_to_region(saved_data, tile, lon_west, lon_east, lat_south, lat_north,
                                                           resolution=resolution)
    return region_lats, region_lons, region_data
