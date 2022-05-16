from read_data_iris import *
import cftime
from datetime import datetime
from datetime import timedelta
from plot_utils import binned_cmap
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from scipy.stats.mstats import pearsonr
from scipy.stats import t as tdist
import pandas as pd
from datetime_utils import *
import warnings
from tqdm import tqdm
import sys


timesteps = 6940
window_size = 7
rolling_window = (
        np.expand_dims(np.arange(window_size), 0) +
        np.expand_dims(np.arange(timesteps-2*(window_size//2)), 0).T
     )

long_window_size = 25
long_rolling_window = (
        np.expand_dims(np.arange(long_window_size), 0) +
        np.expand_dims(np.arange(timesteps-2*(long_window_size//2)), 0).T
     )


def get_slope(data, min_readings=2):
    time_array = np.empty(data.shape)
    for i in range(data.shape[1]):
        time_array[:, i] = i + 1
    time_array[np.isnan(data)] = np.nan
    valid_readings = np.sum(~np.isnan(time_array), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        time_mean = np.expand_dims(np.nanmean(time_array, axis=1), 1)
        data_mean = np.expand_dims(np.nanmean(data, axis=1), 1)
        time_std = np.nanstd(time_array, axis=1)
        data_std = np.nanstd(data, axis=1)
        cov = np.nansum((time_array - time_mean) * (data - data_mean), axis=1) / valid_readings
        cor = cov / (time_std * data_std)
        slope = cov / (time_std ** 2)
    valid_readings = valid_readings.astype(float)
    valid_readings[valid_readings < min_readings] = np.nan
    slope[np.isnan(valid_readings)] = np.nan
    return slope


def get_mean(data):
    data_mean = np.nanmean(data, axis=1)
    return data_mean


def get_corr(data1, data2, min_readings=3, sig_p=1., return_p=False):
    data1[np.isnan(data2)] = np.nan
    data2[np.isnan(data1)] = np.nan
    valid_readings = np.sum(~np.isnan(data1), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        data1_mean = np.expand_dims(np.nanmean(data1, axis=1), 1)
        data2_mean = np.expand_dims(np.nanmean(data2, axis=1), 1)
        data1_std = np.nanstd(data1, axis=1)
        data2_std = np.nanstd(data2, axis=1)
        cov = np.nansum((data1 - data1_mean) * (data2 - data2_mean), axis=1) / valid_readings
        cor = cov / (data1_std * data2_std)
        tstats = cor * np.sqrt(valid_readings - 2) / np.sqrt(1 - cor ** 2)
        p_val = tdist.sf(np.abs(tstats), valid_readings - 2) * 2
    cor[valid_readings < min_readings] = np.nan
    cor[p_val > sig_p] = np.nan
    if return_p:
        return cor, p_val
    else:
        return cor


def filter_tile(lon_west, lon_east, lat_south, lat_north):
    vod = read_data_all_years('VOD', band='X', min_year=2000, max_year=2018,
                                 lon_west=lon_west, lon_east=lon_east,
                                 lat_north=lat_north, lat_south=lat_south)
    sm = read_data_all_years('SM', min_year=2000, max_year=2018,
                             lon_west=lon_west, lon_east=lon_east,
                             lat_north=lat_north, lat_south=lat_south)
    swamps = read_data_all_years('SWAMPS', min_year=2000, max_year=2018,
                                 lon_west=lon_west, lon_east=lon_east,
                                 lat_north=lat_north, lat_south=lat_south)
    vod_data = ma.filled(vod.data, np.nan)
    sm_data = ma.filled(sm.data, np.nan)
    swamps_data = ma.filled(ma.masked_less(swamps.data, 0.), np.nan)
    filtered_vod = np.ones_like(vod_data)*np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for i in range(vod_data.shape[1]):
            for j in range(vod_data.shape[2]):
                if np.any(~np.isnan(vod_data[:, i, j])):
                    filtered_vod[:, i, j] = filter_pixel(vod_data[:, i, j], sm_data[:, i, j], swamps_data[:, i, j])
    ma.set_fill_value(filtered_vod, -999999.0)
    filtered_cube = vod.copy(data=filtered_vod)
    filtered_cube.coord('latitude').bounds = None
    filtered_cube.coord('longitude').bounds = None
    filtered_cube.coord('time').bounds = None
    filtered_cube.units = '1'
    return filtered_cube


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def filter_plot(vod_px, sm_px, swamps_px, event_starts, surface_water_filter, save_filename=False):
    brown = '#E1BE6A'
    green = '#40B0A6'
    save_filename='../figures/pixel_case_studies/india/filter_demo_india_title.pdf'
    fig, host = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(right=0.9)
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par2.spines["right"].set_position(("axes", 1.2))
    make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_position(("axes", 1.8))
    make_patch_spines_invisible(par3)
    par3.spines["right"].set_visible(False)
    par3.set_yticklabels([])
    base_date = datetime(2000, 1, 1)
    pydts = [base_date + timedelta(days=n) for n in range(vod_px.size)]
    start_date = datetime(2010, 5, 1)
    end_date = datetime(2010, 10, 1)
    start_time_idx = np.where(pydts==np.datetime64(start_date))[0][0]
    end_time_idx = np.where(pydts==np.datetime64(end_date))[0][0]
    time_slice = slice(start_time_idx, end_time_idx)
    p4 = par3.bar(pydts[time_slice], surface_water_filter[time_slice], width=1, color='k', edgecolor='k')
    # for e in event_starts:
    #     host.axvline(pydts[e], color='gray')
    p1, = host.plot(pydts[time_slice], vod_px[time_slice],'-o', color=green, ms=3)
    p2, = par1.plot(pydts[time_slice], sm_px[time_slice],'-s', color=brown, ms=4)
    p3, = par2.plot(pydts[time_slice], swamps_px[time_slice],'-x', color='red', ms=5)
    host.set_ylabel("VOD (unitless)", fontsize=16)
    par1.set_ylabel("surface SM (m^3 m^-3)", fontsize=16)
    par2.set_ylabel("surface water (%)", fontsize=16)
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    host.tick_params(labelsize=12)
    par1.tick_params(labelsize=12)
    par2.tick_params(labelsize=12)
    par3.tick_params(labelsize=12)
    par3.set_ylim([0, 1])
    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw) 
    deg = u'\u00B0'
    host.set_title(f'20.125{deg}N, 75.125{deg}E', fontsize=14)
    plt.tight_layout()
    host.set_zorder(2)
    host.set_frame_on(False)
    par1.set_zorder(1)
    par2.set_zorder(1)
    if save_filename:
        plt.savefig(save_filename, dpi=400)
    plt.show()


def filter_pixel(vod_px, sm_px, swamps_px, demo_figure=False):
    vod_slope = np.zeros_like(sm_px)
    sm_slope = np.zeros_like(sm_px)
    smooth_vod = np.ones_like(sm_px)*np.nan
    smooth_swamps = np.ones_like(sm_px)*np.nan
    smoothed_corr = np.ones_like(sm_px)*np.nan
    smooth_vod[window_size//2:vod_px.size-window_size//2]=get_mean(vod_px[rolling_window])
    smooth_swamps[window_size//2:vod_px.size-window_size//2]=get_mean(swamps_px[rolling_window])
    smoothed_corr[long_window_size//2:vod_px.size-long_window_size//2] = get_corr(smooth_vod[long_rolling_window], 
                                                                                  smooth_swamps[long_rolling_window])
    sm_readings_in_week = np.zeros_like(sm_px)*np.nan
    vod_swamps_corr = np.ones_like(sm_px)*np.nan
    vod_slope[window_size//2:vod_px.size-window_size//2] = get_slope(vod_px[rolling_window])
    sm_slope[window_size//2:vod_px.size-window_size//2] = get_slope(sm_px[rolling_window])
    sm_slope_threshold = np.nanpercentile(sm_slope, 85)
    vod_slope_threshold = np.nanpercentile(vod_slope, 15)
    sm_readings_in_week[window_size//2:vod_px.size-window_size//2] = np.sum(~np.isnan(sm_px[rolling_window]), axis=1)
    not_enough_sm  = sm_readings_in_week < 3.
    both_grads = np.logical_and(vod_slope<vod_slope_threshold, np.logical_or(sm_slope>sm_slope_threshold, not_enough_sm))
    both_grads_rolling = ndimage.grey_dilation(both_grads, size=(7,))
    surface_water_filter = both_grads_rolling
    event_ends = np.where(np.diff(surface_water_filter.astype(float))<0.)[0]
    min_7day_corr = ndimage.grey_erosion(smoothed_corr, (7,))
    vod_swamps_uncorr = min_7day_corr > -0.265
    t = np.arange(vod_px.size)
    for event_end in event_ends:
        uncorr_after = np.where(np.logical_and(t>=event_end, vod_swamps_uncorr))[0]
        if uncorr_after.size > 0:
            next_uncorr = uncorr_after[0]
            surface_water_filter[event_end:next_uncorr+3] = True #add 3 to filter to the end of the 7-day window in which corr>0
    vod_filtered = np.copy(vod_px)
    vod_filtered[surface_water_filter] = np.nan
    if demo_figure:
        event_starts = np.where(both_grads)[0]
        filter_plot(vod_px, sm_px, swamps_px, event_starts, surface_water_filter)
    return vod_filtered


if __name__ == '__main__':
    tile_lats_south = [-60, -30, 30]
    tile_lats_north = [-30, 30, 80]
    for lat_south, lat_north in zip(tile_lats_south, tile_lats_north):
        for lon_west in tqdm(np.arange(-180, 151, 30)):
            lon_east = lon_west + 30
            filtered_vod = filter_tile(lon_west, lon_east, lat_south, lat_north)
            lon_tile_label = f'{int(lon_west)}E' if lon_west > 0 else f'{int(-lon_west)}W'
            lat_tile_label = f'{int(lat_south)}N' if lat_south > 0 else f'{int(-lat_south)}S'
            cube_shape = filtered_vod.shape
            iris.save(filtered_vod, f'{save_dir}, VOD-X-band_filtered_surface_water_{lon_tile_label}_{lat_tile_label}.nc',
                      fill_value=-999999.0, chunksizes=(1, cube_shape[1], cube_shape[2]))
