import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import matplotlib.pyplot as plt
from read_data_iris import *
from datetime_utils import *
import sys


def filter_jumps_monthly(vod_tile, allowed_jump_stdevs=2):
    """
    Mask VOD readings that are very different from the readings surrounding them.
    For each month of the year, the mean and standard deviation of the
    absolute difference between consecutive valid readings is computed for each pixel.
    The threshold absolute difference is then (mean + allowed_jump_stdevs * standard deviation).
    Any readings which have a difference above this threshold to BOTH the reading before them
    and the reading after them are masked out.
    """
    # get array of VOD data with dimensions [time, lat, lon], NaNs for missing data
    vod_array = ma.filled(vod_tile.data, np.nan) 
    filtered_vod = np.copy(vod_array)
    # make 1D array with entry for each timestep saying what month it is
    dates = [decimal_year_to_datetime(days_since_1970_to_decimal_year(d)) for d in vod_tile.coord('time').points]
    months = np.array([float(d.month) for d in dates])
    # loop over all pixels
    for lat_idx in range(vod_array.shape[1]):
        for lon_idx in range(vod_array.shape[2]):
            # get VOD time series for pixel
            pixel_vod = vod_array[:, lat_idx, lon_idx]
            # find whether reading at each timestep is valid for that pixel
            valid_readings = ~np.isnan(pixel_vod) 
            # get indices of the timesteps that have valid readings
            valid_readings_idx = np.where(valid_readings)[0] 
            # get array of all the valid readings (just the pixel time series with NaNs removed)
            readings = pixel_vod[valid_readings] 
            # get the month that each valid reading takes place in
            months_readings = months[valid_readings] 
            # compute the absolute values of all the jumps between adjacent valid readings
            abs_jumps = np.abs(np.diff(readings))
            # find the month in which the start point of each jump occurs
            # e.g. if it's the jump from a reading on 30th June to 1st July, will be 6 for June
            start_of_jump_month = months_readings[:-1] 
            # create array for storing the maximum allowed jump between readings in each month
            jump_threshold_months = np.zeros(12,) 
            if readings.size > 0: # pixel has some valid readings, e.g. not ocean
                # create array for storing whether the jumps between each pair of adjacent readings
                # are suspiciously large
                large_jump = np.zeros(readings.size - 1)
                for m in range(12): # loop over all months of year
                    month = m + 1 # so Jan = 1 etc.
                    # get mean and standard deviation of all jumps that start in this month
                    in_month = (start_of_jump_month == month)
                    mean_jump = np.nanmean(abs_jumps[in_month])
                    std_jump = np.nanstd(abs_jumps[in_month])
                    # find maximum allowed jump as mean jump + x * standard deviation of jumps
                    jump_threshold = mean_jump + allowed_jump_stdevs * std_jump
                    jump_threshold_months[m] = jump_threshold
                    # mark jumps as suspiciously large if they are over this value
                    large_jump[in_month] = (abs_jumps[in_month] > jump_threshold)
                # if jump is above the monthly threshold both before and after a timestep,
                # mask out the VOD reading at that timestep
                for t in np.arange(valid_readings_idx.size - 2) + 1:
                    if large_jump[t-1] and large_jump[t]:
                        filtered_vod[valid_readings_idx[t], lat_idx, lon_idx] = np.nan  
    return ma.masked_invalid(filtered_vod)


if __name__ == '__main__':
    save_dir = '/localscratch/wllf029/bethar/filtered_vod'
    tile_lats_south = [-60, -30, 30]
    tile_lats_north = [-30, 30, 80]
    for lat_south, lat_north in zip(tile_lats_south, tile_lats_north):
        for lon_west in tqdm(np.arange(-180, 151, 30)):
            lon_east = lon_west + 30
            vod_tile = read_data_all_years('VOD', band='X', min_year=2000, max_year=2018, 
                                           lon_west=lon_west, lon_east=lon_east, lat_south=lat_south, lat_north=lat_north)
            filtered_vod = filter_jumps_monthly(vod_tile)
            ma.set_fill_value(filtered_vod, -999999.0)
            filtered_cube = vod_tile.copy(data=filtered_vod)
            filtered_cube.coord('latitude').bounds = None
            filtered_cube.coord('longitude').bounds = None
            filtered_cube.coord('time').bounds = None
            filtered_cube.units = '1'
            lon_tile_label = f'{int(lon_west)}E' if lon_west > 0 else f'{int(-lon_west)}W'
            lat_tile_label = f'{int(lat_south)}N' if lat_south > 0 else f'{int(-lat_south)}S'
            cube_shape = filtered_cube.shape
            iris.save(filtered_cube, f'{save_dir}/VOD-X-band_filtered_{lon_tile_label}_{lat_tile_label}.nc',
                      fill_value=-999999.0, chunksizes=(1, cube_shape[1], cube_shape[2]))
