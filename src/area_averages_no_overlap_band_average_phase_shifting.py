"""
Take cross-spectral analysis results produced by csagan_multiprocess.py and compute average coherency across
a specified period band, testing for significance based on the large-scale neighbourhood of each pixel.
Proceed to compute the average period at which significant coherency occurs, the average phase difference
and the width of the 95% confidence interval for the phase difference.

Also contains functions to plot maps of these variables (not included in paper).

Bethan Harris, UKCEH, 18/11/2020
"""

import numpy as np
import time
import pickle
import itertools
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.feature as feat
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from read_csagan_saved_output import read_region_data
from plot_utils import *
from lag_by_land_cover import *


##### CONFIGURATION SECTION #####
##### Edit variables in this section to desired values before running script #####
reference_variable_name = 'IMERG'
response_variable_name = 'VOD_X' # For plot titles/filenames
force_lag = False # e.g. for IMERG/VOD
tile = 'global'
season = 'DJF'
# Path to cross-spectral analysis output (as saved from csagan_multiprocessing.py)
spectra_filename = f"/prj/nceo/bethar/cross_spectral_analysis_results/{tile}_IMERG_VOD_spectra_X_{season}_mask_sw_best85.p"
# Years covered by time series entered into CSA (for computing seasonal mean precipitation)
min_year = 2000
max_year = 2018
# Periods defining the band of variability to analyse
band_days_lower = 25.
band_days_upper = 40.
# Coordinates of bounding box to analyse/plot
lon_west = -180
lon_east = 180
lat_south = -55
lat_north = 55

# Filename and title for plots
base_filename = f'../figures/neighbourhood_average_coherency/no_overlap/{tile}/{reference_variable_name}-{response_variable_name}_{season}_{int(band_days_lower)}-{int(band_days_upper)}_days_noforcelag_sw_best85'
#base_filename = None
title = f'{reference_variable_name}/{response_variable_name} {season}, {int(band_days_lower)}-{int(band_days_upper)} days'
##### End of configuration section #####


def tile_global_validity(season):
    tropics_filename = f'/prj/nceo/bethar/cross_spectral_analysis_results/tropics_IMERG_VOD_spectra_X_{season}_mask_sw_best85.p'
    northern_filename = f'/prj/nceo/bethar/cross_spectral_analysis_results/northern_IMERG_VOD_spectra_X_{season}_mask_sw_best85.p'
    southern_filename = f'/prj/nceo/bethar/cross_spectral_analysis_results/southern_IMERG_VOD_spectra_X_{season}_mask_sw_best85.p'
    _, _, spectra_tropics = read_region_data(tropics_filename, 'tropics', -180, 180, -30, 30)
    _, _, spectra_northern = read_region_data(northern_filename, 'northern', -180, 180, 30, 55)
    _, _, spectra_southern = read_region_data(southern_filename, 'southern', -180, 180, -55, -30)
    no_csa_global = np.zeros((440, 1440), dtype='bool')
    no_csa_global[0:100] = (spectra_southern == {})
    no_csa_global[100:340] = (spectra_tropics == {})
    no_csa_global[340:] = (spectra_northern == {})
    return no_csa_global


def tile_global_from_saved_spectra(season):
    tropics_filename = f'/prj/nceo/bethar/cross_spectral_analysis_results/spectra_nooverlap_tropics_IMERG_VOD_X_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.p'
    northern_filename = f'/prj/nceo/bethar/cross_spectral_analysis_results/spectra_nooverlap_northern_IMERG_VOD_X_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.p'
    southern_filename = f'/prj/nceo/bethar/cross_spectral_analysis_results/spectra_nooverlap_southern_IMERG_VOD_X_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.p'
    spectra_tropics = pickle.load(open(tropics_filename, 'rb'))
    spectra_northern = pickle.load(open(northern_filename, 'rb'))
    spectra_southern = pickle.load(open(southern_filename, 'rb'))
    spectra_global = {}
    for key in spectra_tropics.keys():
        spectra_global[key] = np.empty((440, 1440))
        spectra_global[key][0:100] = spectra_southern[key][20:-20]
        spectra_global[key][100:340] = spectra_tropics[key][20:-20]
        spectra_global[key][340:] = spectra_northern[key][20:-40]
    return spectra_global


if tile != 'global':
    # Check whether you remembered to change the filename for the CSA output... not foolproof...
    variable_name_components = reference_variable_name.split('_') + response_variable_name.split('_')
    if not(all([v in spectra_filename for v in variable_name_components])):
        print('#############################')
        print('Check this is the right CSA output file, it does not seem to match variables selected.')
        print('#############################')
    # Load saved data from cross-spectral analysis
    if 'tropics' in spectra_filename:
        tile = 'tropics'
    elif 'northern' in spectra_filename:
        tile = 'northern'
    elif 'southern' in spectra_filename:
        tile = 'southern'
    elif 'polar' in spectra_filename:
        tile = 'polar'
    lats, lons, spectra = read_region_data(spectra_filename, tile, lon_west, lon_east, lat_south, lat_north, resolution=0.25)
    no_csa = (spectra == {})
else:
    lat_south = -55
    lat_north = 55
    lon_west = -180
    lon_east = 180
    lats = np.arange(lat_south, lat_north, 0.25) + 0.5 * 0.25
    lons = np.arange(lon_west, lon_east, 0.25) + 0.5 * 0.25
    neighbourhood_averages = tile_global_from_saved_spectra(season)
    no_csa = tile_global_validity(season)


def neighbourhood_indices(lat_idx, lon_idx):
    # Get indices for the neighbouring pixels based on centre coordinates (see Harris et al Fig. S2)
    lat_idcs = range(lat_idx-4, lat_idx+5, 4)
    lon_idcs = range(lon_idx-4, lon_idx+5, 4)
    all_pixels = itertools.product(lat_idcs, lon_idcs)
    return all_pixels


def neighbourhood_spectra(spectra_data, lat_idx, lon_idx):
    """
    Turn the array of dictionaries of cross-spectral analysis output into lists of arrays of period and coherency
    for each neighbouring pixel plus central pixel. Each list has one item per pixel in neighbourhood.
    Index names make sense if array has latitude as axis 0, longitude as axis 1 (both increasing with axis index).
    Parameters:
    spectra_data: Array of dictionaries from cross-spectral analysis output
    idx_south (int): Index representing start of neighbourhood on axis 0 (southern boundary). Index included.
    idx_north (int): Index representing end of neighbourhood on axis 0 (northern boundary). Index excluded.
    idx_east (int): Index representing start of neighbourhood on axis 1 (western boundary). Index included.
    idx_west (int): Index representing end of neighbourhood on axis 1 (eastern boundary). Index excluded.
    Returns (lists of 1D arrays):
    List of pixel periods for neighbourhood,
    List of pixel coherencies for neighbourhood (for each period in the period list).
    """
    # Initialise empty lists
    list_of_periods = []
    list_of_coherencies = []
    list_of_phases = []
    list_of_amplitudes = []
    for pixel in neighbourhood_indices(lat_idx, lon_idx): # Loop through each neighbour pixel
        pixel_lat = pixel[0]
        pixel_lon = pixel[1]
        lat_in_bounds = pixel_lat >= 0 and pixel_lat < spectra_data.shape[0]
        lon_in_bounds = pixel_lon >= 0 and pixel_lon < spectra_data.shape[1]
        is_central_pixel = (pixel_lat == lat_idx) and (pixel_lon == lon_idx)
        if lat_in_bounds and lon_in_bounds and not is_central_pixel:
            spectrum = spectra_data[pixel_lat, pixel_lon]
            if spectrum != {}: # Only interested in pixels that had data when cross-spectral analysis was performed
                list_of_periods.append(spectrum['period'][::-1]) # Reverse so periods are increasing
                list_of_coherencies.append(spectrum['coherency'][::-1])
    return list_of_periods, list_of_coherencies


def check_significant_neighbours(spectra_data, lat_idx, lon_idx):
    central_spectra = spectra_data[lat_idx, lon_idx]
    if central_spectra != {}:
        central_periods = central_spectra['period'][::-1]
        central_coherencies = central_spectra['coherency'][::-1]
        neighbour_periods, neighbour_coherencies = neighbourhood_spectra(spectra_data, lat_idx, lon_idx)
        central_period_band = np.logical_and(central_periods<=band_days_upper, central_periods>=band_days_lower)
        min_period_gap = np.diff(central_periods[central_period_band]).min()
        central_sig_periods = central_coherencies > 0.7795
        central_sig_periods_in_band = central_periods[np.logical_and(central_period_band, central_sig_periods)]
        significant_neighbours = np.zeros_like(central_sig_periods_in_band)
        resolution_bandwidth = central_spectra['resolution_bandwidth']
        min_periods = 1./((1./central_sig_periods_in_band) + 0.5*resolution_bandwidth)
        max_periods = 1./((1./central_sig_periods_in_band) - 0.5*resolution_bandwidth)
        for p, c in zip(neighbour_periods, neighbour_coherencies):
            for i, test_period in enumerate(central_sig_periods_in_band):
                period_within_rbw = np.logical_and(p<=max_periods[i], p>=min_periods[i])
                coh_sig_within_rbw = np.logical_and(period_within_rbw, c>0.7795)
                significant_neighbours[i] += int(np.any(coh_sig_within_rbw))
        significant_periods = central_sig_periods_in_band[significant_neighbours>2.]
        significant_idcs = np.isin(central_periods, significant_periods)
        central_phases = central_spectra['phase'][::-1]
        central_amplitudes = central_spectra['amplitude'][::-1]
        return central_periods[significant_idcs], central_phases[significant_idcs], central_coherencies[significant_idcs], central_amplitudes[significant_idcs]
    else:
        return None


def compute_phase_error(coherency):
    return_float = False
    if isinstance(coherency, float):
        coherency = np.array([coherency])
        return_float = True
    error_coeff = 0.238 / (2. * (1. - 0.238))
    phase_error = np.sqrt(error_coeff * (1./coherency**2 - 1.))
    final_errors = np.empty_like(phase_error)
    within_90 = (2.447 * phase_error) < 1.
    final_errors[within_90] = np.arcsin(2.447 * phase_error[within_90])
    final_errors[~within_90] = np.maximum(np.arcsin(1.), 1.96*np.sqrt(0.238)*np.sqrt(0.5 * (1./coherency[~within_90]**2 - 1.)))
    final_errors_degrees = 180. * final_errors / np.pi
    if return_float:
        return final_errors_degrees[0]
    else:
        return final_errors_degrees


def average_intraseasonal_coherency(coords):
    lat, lon = coords
    if spectra[lat, lon] != {}:
        try:
            nbhd_test = check_significant_neighbours(spectra, lat, lon)
            if nbhd_test is not None:
                sig_periods = nbhd_test[0]
                sig_phases = nbhd_test[1]
                sig_coherencies = nbhd_test[2]
                sig_amplitudes = nbhd_test[3]
                if sig_periods.size > 1:
                    avg_coherency = np.mean(sig_coherencies)
                    avg_period = np.mean(sig_periods)
                    if np.any(np.abs(sig_phases) > 150.):
                        sig_phases[sig_phases<0.] += 360.
                    avg_phase = np.mean(sig_phases)
                    if avg_phase > 180.:
                        avg_phase -= 360.
                    weighted_avg_phase = np.average(sig_phases, weights=sig_periods)
                    weighted_avg_coh = np.average(sig_coherencies, weights=sig_periods)
                    avg_lag = weighted_avg_phase / 360. * avg_period
                    if avg_lag > 0.5*avg_period:
                        avg_lag -= avg_period
                    avg_amplitude = np.mean(sig_amplitudes)
                    phase_error = compute_phase_error(avg_coherency)
                    lag_error = compute_phase_error(weighted_avg_coh) / 360. * avg_period
                else:
                    avg_coherency = sig_coherencies[0]
                    avg_period = sig_periods[0]
                    avg_phase = sig_phases[0]
                    avg_lag = avg_phase / 360. * avg_period
                    avg_amplitude = sig_amplitudes[0]
                    phase_error = compute_phase_error(avg_coherency)
                    lag_error = phase_error / 360. * avg_period
            else:
                avg_coherency = np.nan
                avg_period = np.nan
                avg_phase = np.nan
                avg_lag = np.nan
                avg_amplitude = np.nan
                lag_error = np.nan
        except: # Some pixels might not sample enough frequencies for the period band
            avg_coherency = np.nan
            avg_period = np.nan
            avg_phase = np.nan
            avg_lag = np.nan
            avg_amplitude = np.nan
            lag_error = np.nan
    else:
        avg_coherency = np.nan
        avg_period = np.nan
        avg_phase = np.nan
        avg_lag = np.nan
        avg_amplitude = np.nan
        lag_error = np.nan
    return avg_coherency, avg_period, avg_phase, avg_lag, lag_error, avg_amplitude


def plot_map_global(data, cmap, norm, cbar_ticks, colorbar_label, title, save_filename=None):
    lon_bounds = np.hstack((lons - 0.5*0.25, np.array([lons[-1]+0.5*0.25])))
    lat_bounds = np.hstack((lats - 0.5*0.25, np.array([lats[-1]+0.5*0.25])))
    projection = ccrs.Robinson()
    fig = plt.figure(figsize=(15, 5)) 
    ax = plt.axes(projection=projection)
    p = plt.pcolormesh(lon_bounds, lat_bounds, data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
    plt.title(title, fontsize=14)
    ax.set_extent([lon_west, lon_east, lat_south, lat_north], crs=ccrs.PlateCarree())
    ax.add_feature(feat.BORDERS, linestyle=':', linewidth=1)
    cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.12, ax.get_position().width, 0.03])
    cbar = fig.colorbar(p, orientation='horizontal', cax=cax, aspect=40, pad=0.01)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_xlabel(colorbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    ax.coastlines(color='black', linewidth=1)
    ax.gridlines(draw_labels=True, color='k', alpha=0.1)
    if save_filename:
        plt.savefig(f'{save_filename}.png', dpi=600, bbox_inches='tight', pad_inches=0.1)


def plot_map_tile(data, cmap, norm, cbar_ticks, colorbar_label, title, save_filename=None, resolution=0.25):
    lon_bounds = np.hstack((lons - 0.5*resolution, np.array([lons[-1]+0.5*resolution])))
    lat_bounds = np.hstack((lats - 0.5*resolution, np.array([lats[-1]+0.5*resolution])))
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(15, 5)) 
    ax = plt.axes(projection=projection)
    p = plt.pcolormesh(lon_bounds, lat_bounds, data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
    plt.title(title, fontsize=14)
    north_offset = 0 if tile=='polar' else -5
    ax.set_extent((lon_west, lon_east, lat_south+5, lat_north+north_offset), crs=projection)
    cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.12, ax.get_position().width, 0.03])
    cbar = fig.colorbar(p, orientation='horizontal', cax=cax, aspect=40, pad=0.12)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_xlabel(colorbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    ax.coastlines(color='black', linewidth=1)
    ax.set_xticks(np.arange(lon_west, lon_east+1, 90), crs=projection)
    north_tick_offset = 1 if tile=='polar' else -4
    ax.set_yticks(np.arange(lat_south+5, lat_north+north_tick_offset, 30), crs=projection)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=16)
    ax.tick_params(axis='x', pad=5)
    if save_filename:
        plt.savefig(f'{save_filename}.png', dpi=600, bbox_inches='tight', pad_inches=0.1)


def plot_map(data, cmap, norm, cbar_ticks, colorbar_label, title, extend='neither', save_filename=None):
    tropics_tile = (lat_south==-35 and lat_north==35) and (lon_west==-180 and lon_east==180)
    northern_tile = (lat_south==25 and lat_north==65) and (lon_west==-180 and lon_east==180)
    southern_tile = (lat_south==-60 and lat_north==-25) and (lon_west==-180 and lon_east==180)
    polar_tile = (lat_south==55 and lat_north==80) and (lon_west==-180 and lon_east==180)
    if tropics_tile or northern_tile or southern_tile or polar_tile:
        plot_map_tile(data, cmap, norm, cbar_ticks, colorbar_label, title, save_filename=save_filename)
    elif tile == 'global':
        plot_map_global(data, cmap, norm, cbar_ticks, colorbar_label, title, save_filename=save_filename)
    else:
        states_provinces = feat.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='10m',
            facecolor='none')
        projection = ccrs.PlateCarree()
        fig = plt.figure() 
        ax = plt.axes(projection=projection)
        p = plt.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
        plt.title(title, fontsize=14)
        ax.set_extent((lon_west, lon_east, lat_south, lat_north), crs=projection)
        cax = fig.add_axes([ax.get_position().x1+0.05,ax.get_position().y0, 0.02, ax.get_position().height])
        cbar = fig.colorbar(p, orientation='vertical', cax=cax, aspect=40, pad=0.12, extend=extend)
        cbar.set_ticks(cbar_ticks)
        cbar.ax.set_ylabel(colorbar_label, fontsize=14)
        cbar.ax.tick_params(labelsize=14)
        ax.coastlines(color='black', linewidth=1)
        ax.add_feature(feat.BORDERS, linestyle=':', linewidth=1)
        ax.set_xticks(np.arange(lon_west, lon_east+1, 10), crs=projection)
        ax.set_yticks(np.arange(lat_south, lat_north+1, 10), crs=projection)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize=16)
        ax.tick_params(axis='x', pad=5)
        if save_filename:
            plt.savefig(f'{save_filename}.png', dpi=600, bbox_inches='tight', pad_inches=0.1)


def plot_coherency_map(mask_coherency, title, save_filename):
    cmap = mpl.cm.get_cmap("viridis").copy()
    norm = mpl.colors.Normalize(vmin=0.7795, vmax=1.)
    cmap.set_bad('#dddddd')
    cmap.set_under('white')
    invalid_but_csa = np.logical_and(~no_csa, np.isnan(mask_coherency))
    mask_coherency[invalid_but_csa] = -999
    plot_map(mask_coherency, cmap, norm, [0.8, 0.9, 1.], 'coherency', title, save_filename=save_filename)
    

def plot_period_map(mask_period, title, save_filename):
    band_width = band_days_upper - band_days_lower
    if band_width <= 10:
        period_increment = 1
    elif 10 < band_width and band_width <= 20:
        period_increment = 2
    elif 20 < band_width and band_width <= 50:
        period_increment = 5
    else:
        period_increment = 10
    period_contour_levels = np.arange(band_days_lower, band_days_upper+1, period_increment).astype(int)
    cmap, norm = binned_cmap(period_contour_levels, 'plasma')
    cmap.set_bad('#dddddd')
    cmap.set_under('white')
    invalid_but_csa = np.logical_and(~no_csa, np.isnan(mask_period))
    mask_period[invalid_but_csa] = -999
    cbar_label = 'period (days)'
    plot_map(mask_period, cmap, norm, period_contour_levels, cbar_label, title, save_filename=save_filename)


def plot_lag_map(mask_lag, title, save_filename):
    contour_multiple = 5.
    lag_contour_levels = np.arange(-0.5*band_days_upper, 0.5*band_days_upper + 1., contour_multiple).astype(int)
    colormap = cm.get_cmap('Spectral', lag_contour_levels.size+1)
    colors = list(colormap(np.arange(lag_contour_levels.size+1)))
    colors_each_side = (lag_contour_levels.size - 1)//2
    no_centre_colors = colors[0:colors_each_side] + colors[-colors_each_side:]
    cmap = mpl.colors.ListedColormap(no_centre_colors, "")
    norm = mpl.colors.BoundaryNorm(lag_contour_levels, ncolors=len(lag_contour_levels)-1, clip=False)
    cmap.set_bad('#dddddd')
    cmap.set_under('white')
    invalid_but_csa = np.logical_and(~no_csa, np.isnan(mask_lag))
    mask_lag[invalid_but_csa] = -999
    cbar_label = 'phase difference (days)'
    plot_map(mask_lag, cmap, norm, lag_contour_levels, cbar_label, title, save_filename=save_filename)


def plot_phase_map(mask_phase, title, save_filename):
    phase_contour_levels = np.arange(-180, 181, 30)
    colormap = cm.get_cmap('coolwarm', phase_contour_levels.size+3)
    colors = list(colormap(np.arange(phase_contour_levels.size+3)))
    colors_each_side = (phase_contour_levels.size - 1)//2
    no_centre_colors = colors[0:colors_each_side] + colors[-colors_each_side:]
    cmap = mpl.colors.ListedColormap(no_centre_colors, "")
    norm = mpl.colors.BoundaryNorm(phase_contour_levels, ncolors=len(phase_contour_levels)-1, clip=False)
    cmap.set_bad('#dddddd')
    cmap.set_under('white')
    invalid_but_csa = np.logical_and(~no_csa, np.isnan(mask_phase))
    mask_phase[invalid_but_csa] = -999
    cbar_label = 'phase difference (deg)'
    plot_map(mask_phase, cmap, norm, phase_contour_levels, cbar_label, title, save_filename=save_filename)


def plot_lag_error_map(mask_lag_error, title, save_filename):
    try:
        lag_error_contour_levels = np.arange(0, np.nanmax(mask_lag_error) + 1.).astype(int)
    except:
        lag_error_contour_levels = np.arange(0, 8)
    cmap, norm = binned_cmap(lag_error_contour_levels, 'magma')
    cmap.set_bad('#dddddd')
    cmap.set_under('white')
    invalid_but_csa = np.logical_and(~no_csa, np.isnan(mask_lag_error))
    mask_lag_error[invalid_but_csa] = -999
    cbar_label = 'phase difference error (days)'
    plot_map(mask_lag_error, cmap, norm, lag_error_contour_levels, cbar_label, title, save_filename=save_filename)


def plot_lag_histogram(mask_lag, title, save_filename=None):
    mask_lag[mask_lag==-999] = np.nan
    contour_multiple = 5.
    try:
        lower_lag_bound = contour_multiple * np.floor(float(np.nanmin(mask_lag))/contour_multiple)
        upper_lag_bound = contour_multiple * np.ceil(float(np.nanmax(mask_lag))/contour_multiple)
        lag_contour_levels = np.arange(lower_lag_bound, upper_lag_bound + 1., contour_multiple).astype(int)
    except:
        lag_contour_levels = np.arange(0, band_days_upper, contour_multiple)
    plt.figure()
    plt.hist(mask_lag.ravel(), bins=lag_contour_levels, edgecolor='k', facecolor='#3ec966')
    plt.xlabel('lag (days)', fontsize=16)
    plt.xticks(lag_contour_levels)
    plt.ylabel('number of pixels', fontsize=16)
    plt.title(title, fontsize=14)
    plt.gca().tick_params(labelsize=14)
    plt.tight_layout()
    if save_filename:
        plt.savefig(f'{save_filename}.png', dpi=600, bbox_inches='tight', pad_inches=0.1)


def plot_amplitude_map(mask_amplitude, title, save_filename):
    cmap = mpl.cm.get_cmap("YlGn").copy()
    norm = mpl.colors.Normalize(vmin=0., vmax=0.05)
    cbar_ticks = np.linspace(0., 0.05, 6, endpoint=True)
    cmap.set_bad('#dddddd')
    cmap.set_under('white')
    invalid_but_csa = np.logical_and(~no_csa, np.isnan(mask_amplitude))
    mask_amplitude[invalid_but_csa] = -999
    plot_map(mask_amplitude, cmap, norm, cbar_ticks, 'amplitude', title, extend='max', save_filename=save_filename)


def mask_to_95pc_coherency_confidence(neighbourhood_averages):
    masked_neighbourhood_averages = {}
    coherency = np.copy(neighbourhood_averages['coherency'])
    for key in neighbourhood_averages.keys():
        masked_average = np.copy(neighbourhood_averages[key])
        masked_average[coherency < 0.7795] = np.nan
        masked_neighbourhood_averages[key] = masked_average
    return masked_neighbourhood_averages


def plot_all_analysis(neighbourhood_averages, title, base_filename):
    masked_neighbourhood_averages = mask_to_95pc_coherency_confidence(neighbourhood_averages)
    plot_coherency_map(masked_neighbourhood_averages['coherency'], title, base_filename)
    plot_period_map(masked_neighbourhood_averages['period'], title, f'{base_filename}_period')
    plot_lag_map(masked_neighbourhood_averages['lag'], title, f'{base_filename}_lag')
    plot_phase_map(masked_neighbourhood_averages['phase'], title, f'{base_filename}_phase')
    plot_lag_error_map(masked_neighbourhood_averages['lag_error'], title, f'{base_filename}_lag_error')
    plot_lag_histogram(masked_neighbourhood_averages['lag'], title, f'{base_filename}_lag_histogram')
    plot_amplitude_map(masked_neighbourhood_averages['amplitude'], title, f'{base_filename}_amplitude')
    plt.show()


def run_neighbourhood_averaging():
    lat_idcs = np.arange(lats.size)
    lon_idcs = np.arange(lons.size)
    LAT, LON = np.meshgrid(lat_idcs, lon_idcs)
    coords = zip(LAT.ravel(), LON.ravel())
    neighbourhood_averages = {}
    print(f'Start averaging {lats.size * lons.size} pixels')
    start = time.time()
    with Pool(processes=4) as pool:
        output = pool.map(average_intraseasonal_coherency, coords, chunksize=1)
    output_array = np.array(output)
    neighbourhood_averages['coherency'] = np.reshape(output_array[:, 0], (lats.size, lons.size), order='F')
    neighbourhood_averages['period'] = np.reshape(output_array[:, 1], (lats.size, lons.size), order='F')
    neighbourhood_averages['phase'] = np.reshape(output_array[:, 2], (lats.size, lons.size), order='F')
    neighbourhood_averages['lag'] = np.reshape(output_array[:, 3], (lats.size, lons.size), order='F')
    neighbourhood_averages['lag_error'] = np.reshape(output_array[:, 4], (lats.size, lons.size), order='F')
    neighbourhood_averages['amplitude'] = np.reshape(output_array[:, 5], (lats.size, lons.size), order='F')
    end = time.time()
    print(f'Time taken to compute neighbourhood averages: {end-start:0.1f} seconds')
    return neighbourhood_averages


def lag_sign_stats(neighbourhood_averages):
    masked_neighbourhood_averages = mask_to_95pc_coherency_confidence(neighbourhood_averages)
    lag50 = masked_neighbourhood_averages['lag']
    lag_error50 = masked_neighbourhood_averages['lag_error']
    lag_lower=lag50 - lag_error50
    lag_upper=lag50 + lag_error50
    total_px = (~np.isnan(lag50)).sum()
    pos_px = (lag50 > 0.).sum()
    neg_px = (lag50 < 0.).sum()
    pos_less_7 = np.logical_and(lag50>0., lag50<7.).sum()
    pos_less_10 = np.logical_and(lag50>0., lag50<10.).sum()
    pos_ci_px = (lag_lower>0.).sum()
    neg_ci_px = (lag_upper<0.).sum()
    cross_ci = np.logical_and(lag_upper>0., lag_lower<0.)
    cross_ci_px = (cross_ci).sum()
    cross_ci_px_pos = (np.logical_and(cross_ci, lag50>0.)).sum()
    cross_ci_px_neg = (np.logical_and(cross_ci, lag50<0.)).sum()
    print(f'total pixels: {total_px}')
    print(f'positive lag: {pos_px}')
    print(f'negative lag: {neg_px}')
    print(f'positive and less than 7: {pos_less_7}')
    print(f'positive and less than 10: {pos_less_10}')
    print('Accounting for 95% CI:')
    print(f'positive: {pos_ci_px}')
    print(f'negative: {neg_ci_px}')
    print(f'sign uncertain: {cross_ci_px}')
    print(f'of which mean lag positive: {cross_ci_px_pos}')
    print(f'of which mean lag negative: {cross_ci_px_neg}')


if __name__ == '__main__':
    neighbourhood_averages = run_neighbourhood_averaging()
    pickle.dump(neighbourhood_averages, open(f'/prj/nceo/bethar/cross_spectral_analysis_results/spectra_nooverlap_{tile}_IMERG_VOD_X_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.p', 'wb'))

