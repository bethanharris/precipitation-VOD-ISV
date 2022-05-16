import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator
from dateutil.relativedelta import relativedelta
from brokenaxes import brokenaxes
from datetime import datetime, timedelta
import cartopy
import cartopy.feature as feat
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from bandpass_filters import *
from read_csagan_saved_output import read_region_data


start_dates = {'australia_3dlagDJFnonzero': (datetime(2007, 12, 1), datetime(2010, 12, 1), datetime(2013,12,1)),
               'east_africa_20dlagMAM': (datetime(2013, 3, 1), datetime(2014, 3, 1), datetime(2015, 3, 1)),
               'madagascar_-22dlagMAM': (datetime(2009, 3, 1), datetime(2010, 3, 1), datetime(2014, 3, 1))}

end_dates = {'australia_3dlagDJFnonzero': (datetime(2008, 3, 1), datetime(2011, 3, 1), datetime(2014,3,1)),
             'east_africa_20dlagMAM': (datetime(2013, 6, 1), datetime(2014, 6, 1), datetime(2015, 6, 1)),
             'madagascar_-22dlagMAM': (datetime(2009, 6, 1), datetime(2010, 6, 1), datetime(2014, 6, 1))}

px_lats = {'australia_3dlagDJFnonzero': -24.625,
           'east_africa_20dlagMAM': 3.875,
           'madagascar_-22dlagMAM': -18.625}

px_lons = {'australia_3dlagDJFnonzero': 125.375,
           'east_africa_20dlagMAM': 31.875,
           'india_swtestMAM': 76.125}

px_period_bands = {'australia_3dlagDJFnonzero': 'lower',
                   'east_africa_20dlagMAM': 'upper',
                   'madagascar_-22dlagMAM': 'lower'}

px_seasons = {'australia_3dlagDJFnonzero': 'DJF',
              'east_africa_20dlagMAM': 'MAM',
              'madagascar_-22dlagMAM': 'MAM'}

vod_colour ='#F93434'


def pixel_lag_data(px_desc):
    lats = np.arange(-60,80,0.25) + 0.5*0.25
    lons = np.arange(-180,180,0.25) + 0.5*0.25
    lat_idx = np.where(lats==px_lats[px_desc])[0][0]
    lon_idx = np.where(lons==px_lons[px_desc])[0][0]
    season = px_seasons[px_desc]
    band = px_period_bands[px_desc]
    if band == 'lower':
        lower = 25
        upper = 40
    elif band == 'upper':
        lower = 40
        upper = 60
    else:
        raise KeyError('Period band must be lower (25-40) or upper (40-60)')
    lags = np.load(f'../data/lag_subplots_data/lag_{season}_{int(lower)}-{int(upper)}.npy')
    lag_errors = np.load(f'../data/lag_subplots_data/lag_error_{season}_{int(lower)}-{int(upper)}.npy')
    periods = np.load(f'../data/lag_subplots_data/period_{season}_{int(lower)}-{int(upper)}.npy')
    lag_data = {'lag': lags[lat_idx, lon_idx], 'lag_error': lag_errors[lat_idx,lon_idx], 'period': periods[lat_idx,lon_idx]}
    return lag_data


def read_time_series(px_desc):
    save_directory = f'../data/pixel_time_series/{px_desc}'
    imerg_anom = np.load(f'{save_directory}/imerg_anom_{px_desc}.npy')
    vod_anom = np.load(f'{save_directory}/vod_anom_{px_desc}.npy')
    base_date = datetime(2000, 1, 1)
    dates = [base_date + timedelta(days=n) for n in range(vod_anom.size)]
    return dates, imerg_anom, vod_anom


def time_series_mask_dates(px_desc):
    dates, imerg_anom, vod_anom = read_time_series(px_desc)
    start_date_list = start_dates[px_desc]
    end_date_list = end_dates[px_desc]
    all_imerg = np.ones_like(imerg_anom)*np.nan
    all_vod = np.ones_like(vod_anom)*np.nan
    if not isinstance(start_date_list, tuple):
        start_date_idx = np.where(dates==np.datetime64(start_date_list))[0][0]
        end_date_idx = np.where(dates==np.datetime64(end_date_list))[0][0]
        time_slice = slice(start_date_idx, end_date_idx)
        all_imerg[time_slice] = imerg_anom[time_slice]
        all_vod[time_slice] = vod_anom[time_slice]
        return dates, all_imerg, all_vod
    else:
        for season in range(len(start_date_list)):
            start_date_idx = np.where(dates==np.datetime64(start_date_list[season]))[0][0]
            end_date_idx = np.where(dates==np.datetime64(end_date_list[season]))[0][0]
            season_time_slice = slice(start_date_idx, end_date_idx)
            all_imerg[season_time_slice] = imerg_anom[season_time_slice]
            all_vod[season_time_slice] = vod_anom[season_time_slice]
        return dates, all_imerg, all_vod


def filter_imerg_seasons(px_desc, all_imerg):
    period_band = px_period_bands[px_desc]
    if period_band == 'lower':
        filtered_imerg = bandpass_filter_missing_data(all_imerg, 1./40., 1./25., 
                                                      1., order=1, min_slice_size=60)
    elif period_band == 'upper':
        filtered_imerg = bandpass_filter_missing_data(all_imerg, 1./60., 1./40., 
                                                      1., order=1, min_slice_size=60)
    else:
        raise KeyError('Period band must be lower (25-40) or upper (40-60)')
    return filtered_imerg


def filter_vod_seasons(px_desc, dates, all_imerg, all_vod, window_size=5):
    dates = np.array(dates)
    missing_data = np.isnan(all_imerg)
    change_missing = np.diff(missing_data.astype(float))
    start_valid = (np.where(change_missing == -1)[0] + 1).tolist()
    end_valid = (np.where(change_missing == 1)[0] + 1).tolist()
    if not missing_data[0]:
        start_valid.insert(0, 0)
    if len(end_valid) == len(start_valid) - 1:
        end_valid.append(data.size)
    valid_data_slices = zip(start_valid, end_valid)
    filtered_dates = np.array([])
    filtered_vod = np.array([])
    for start, end in valid_data_slices:
        vod_slice = all_vod[start:end]
        valid_vod_obs_slice = ~np.isnan(vod_slice)
        slice_idcs = np.arange(start, end)
        valid_obs_in_bins = np.sum(np.pad(valid_vod_obs_slice.astype(int), 
                                   (0, ((window_size - slice_idcs.size%window_size) % window_size)), 
                                   mode='constant', constant_values=0).reshape(-1, window_size), 
                                   axis=1).astype(int)
        date_idcs = np.nanmean(np.pad(slice_idcs.astype(float), 
                                  (0, ((window_size - slice_idcs.size%window_size) % window_size)), 
                                   mode='constant', constant_values=np.nan).reshape(-1, window_size), 
                                   axis=1).astype(int)
        binned_means = np.nanmean(np.pad(vod_slice.astype(float), 
                                  (0, ((window_size - vod_slice.size%window_size) % window_size)), 
                                   mode='constant', constant_values=np.nan).reshape(-1, window_size), 
                                   axis=1)
        binned_means[valid_obs_in_bins < 2] = np.nan
        dummy_date = dates[date_idcs[-1]] + timedelta(days=1) # so the separate seasons don't join up on line plots
        filtered_dates = np.hstack((filtered_dates, dates[date_idcs], np.array([dummy_date])))
        filtered_vod = np.hstack((filtered_vod, binned_means, np.array([np.nan])))
    return filtered_dates, filtered_vod


def plot_time_series_multiyear(px_desc, ax_to_plot, label_letter):
    dates, imerg, vod = time_series_mask_dates(px_desc)
    axis_start_points = start_dates[px_desc]
    axis_end_points = end_dates[px_desc]
    bax_xlims = tuple([(s, e) for s, e in zip(axis_start_points, axis_end_points)])
    imerg_limit = np.nanmax(np.abs(imerg)) * 1.1
    vod_limit = np.nanmax(np.abs(vod)) * 1.1
    bax = brokenaxes(subplot_spec=ax_to_plot, xlims=bax_xlims)
    bax.big_ax.axhline(0.5, color='gray', linestyle='--', zorder=0)
    bax.plot(dates, imerg, 'k-o', ms=2, zorder=1)
    bax.plot(dates, imerg, 'k',alpha=0.5,zorder=1)
    [x.remove() for x in bax.diag_handles]
    bax.draw_diags()
    bax.big_ax.set_ylabel('precipitation\nanomaly (mm day$^{-1}$)', fontsize=12, labelpad=30)
    for i, ax in enumerate(bax.axs):
        ax.tick_params(labelsize=14)
        ax.set_ylim([-imerg_limit, imerg_limit])
        ax.set_xticks([axis_start_points[i], axis_start_points[i]+relativedelta(months=2)])
        ax.tick_params(axis='x', rotation=7)
    ax2 = brokenaxes(subplot_spec=ax_to_plot, xlims=bax_xlims)
    ax2.plot(dates, vod, '-s', ms=2, color=vod_colour)
    for ax in ax2.axs:
        ax.patch.set_facecolor('none')
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(True)
        ax.spines['top'].set_color('#cccccc')
        ax.tick_params(labelsize=14)
        ax.tick_params(left=False, labelleft=False)
        ax.tick_params(bottom=False, labelbottom=False)
        ax.set_ylim([-vod_limit, vod_limit])
    ax2.axs[-1].spines['right'].set_visible(True)
    ax2.axs[-1].tick_params(right=True, labelright=True)
    ax2.axs[-1].tick_params(axis='y', colors=vod_colour)
    ax2.big_ax.yaxis.set_label_position("right")
    ax2.big_ax.set_ylabel('VOD anomaly\n(unitless)', fontsize=12, labelpad=50, color=vod_colour)
    px_lag_data = pixel_lag_data(px_desc)
    px_lag = px_lag_data['lag']
    px_lag_error = px_lag_data['lag_error']
    px_period = px_lag_data['period']
    px_lag_label = f'+{px_lag:.1f}' if px_lag>0 else f'{px_lag:.1f}'
    lag_summary = f'{px_seasons[px_desc]} phase diff.: {px_lag_label} $\pm$ {px_lag_error:.1f} days @ {px_period:.1f} day period'
    ax2.big_ax.text(0.99, 0.05, lag_summary, transform=ax.transAxes, fontsize=12, 
                    color='#8C8888', ha='right', bbox=dict(facecolor='white', 
                    edgecolor='none', pad=1.0))
    deg = u'\u00B0'
    bax.big_ax.set_title(f'({label_letter}) {px_lats[px_desc]}{deg}N, {px_lons[px_desc]}{deg}E', fontsize=14, color='k')


def plot_time_series_multiyear_filtered(px_desc, ax_to_plot, label_letter, window_size=5):
    dates, imerg, vod = time_series_mask_dates(px_desc)
    filtered_imerg = filter_imerg_seasons(px_desc, imerg)
    filtered_dates, filtered_vod = filter_vod_seasons(px_desc, dates, imerg, vod, 
                                                      window_size=window_size)
    axis_start_points = start_dates[px_desc]
    axis_end_points = end_dates[px_desc]
    bax_xlims = tuple([(s, e) for s, e in zip(axis_start_points, axis_end_points)])
    imerg_limit = np.nanmax(np.abs(filtered_imerg)) * 1.1
    vod_limit = np.nanmax(np.abs(vod)) * 1.1
    bax = brokenaxes(subplot_spec=ax_to_plot, xlims=bax_xlims)
    zero_line = np.zeros_like(filtered_imerg)
    bax.plot(dates, zero_line, color='#8C8888', linestyle='-', linewidth=0.75, zorder=0)
    bax.plot(dates, filtered_imerg, 'k',alpha=1,zorder=1)
    [x.remove() for x in bax.diag_handles]
    bax.draw_diags()
    bax.big_ax.set_ylabel('precipitation anomaly\n(mm day$^{-1}$)', fontsize=12, labelpad=30)
    for i, ax in enumerate(bax.axs):
        ax.tick_params(labelsize=14)
        ax.set_ylim([-imerg_limit, imerg_limit])
        ax.set_xticks([axis_start_points[i], axis_start_points[i]+relativedelta(months=2)])
        ax.tick_params(axis='x', rotation=7)
    ax2 = brokenaxes(subplot_spec=ax_to_plot, xlims=bax_xlims)
    ax2.scatter(dates, vod, s=2, c=vod_colour, alpha=0.5)
    ax2.plot(filtered_dates, filtered_vod, '-x', color=vod_colour, ms=2)
    for ax in ax2.axs:
        ax.patch.set_facecolor('none')
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(True)
        ax.spines['top'].set_color('#cccccc')
        ax.tick_params(labelsize=14)
        ax.tick_params(left=False, labelleft=False)
        ax.tick_params(bottom=False, labelbottom=False)
        ax.set_ylim([-vod_limit, vod_limit])
    ax2.axs[-1].spines['right'].set_visible(True)
    ax2.axs[-1].tick_params(right=True, labelright=True)
    ax2.axs[-1].tick_params(axis='y', colors=vod_colour)
    ax2.big_ax.yaxis.set_label_position("right")
    ax2.big_ax.set_ylabel('VOD anomaly\n(unitless)', fontsize=12, labelpad=52, color=vod_colour)
    px_lag_data = pixel_lag_data(px_desc)
    px_lag = px_lag_data['lag']
    px_lag_error = px_lag_data['lag_error']
    px_period = px_lag_data['period']
    px_lag_label = f'+{px_lag:.1f}' if px_lag>0 else f'{px_lag:.1f}'
    lag_summary = f'{px_seasons[px_desc]} phase diff.: {px_lag_label} $\pm$ {px_lag_error:.1f} days @ {px_period:.1f} day period'
    ax2.big_ax.text(0.99, 0.05, lag_summary, transform=ax.transAxes, fontsize=12, 
                    color='#8C8888', ha='right', bbox=dict(facecolor='white', 
                    edgecolor='none', pad=0.5))
    deg = u'\u00B0'
    bax.big_ax.set_title(f'$\\bf{{({label_letter})}}$ {px_lats[px_desc]}{deg}N, {px_lons[px_desc]}{deg}E', fontsize=14, color='k')


def map_of_pixels(pixels_to_plot, ax):
    ax.set_extent((-180, 180, -55, 55), crs=ccrs.PlateCarree())
    ax.coastlines(color='black', linewidth=0.5, zorder=1)
    letter_labels = [f'({letter})' for letter in string.ascii_lowercase[1:len(pixels_to_plot)+1]]
    px_labels = {px: letter_label for px, letter_label in zip(pixels_to_plot, letter_labels)}
    for px in pixels_to_plot:
        ax.scatter(px_lons[px], px_lats[px], s=10, c='k', transform=ccrs.PlateCarree(), zorder=3)
        ax.annotate(px_labels[px], (px_lons[px]+1, px_lats[px]+4), color='k',
                    transform=ccrs.PlateCarree(), ha='center', va='bottom',
                    bbox=dict(boxstyle="circle,pad=0.0", fc="white", ec="none", lw=0.75), zorder=2)
    ax.set_xticks(np.arange(-90, 91, 90), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-50, 51, 50), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=14)
    ax.tick_params(axis='x', pad=5)
    ax.set_title('$\\bf{(a)}$ Example time series locations', fontsize=14) 


def all_pixels(window_size=7):
    pixels_to_plot = ['east_africa_20dlagMAM', 'madagascar_-22dlagMAM', 'australia_3dlagDJFnonzero']
    number_pixels = len(pixels_to_plot)
    alphabet = string.ascii_lowercase[1:number_pixels+1]
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(number_pixels+1, 1, hspace=0.6)
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    map_of_pixels(pixels_to_plot, ax_map)
    i = 1
    for pixel_desc, pixel_label in zip(pixels_to_plot, alphabet):
        plot_time_series_multiyear_filtered(pixel_desc, gs[i, 0], pixel_label, window_size=window_size)
        i += 1
    plt.savefig(f'../figures/example_pixel_time_series_filtered_bin{int(window_size)}.pdf', bbox_inches='tight')
    plt.savefig(f'../figures/example_pixel_time_series_filtered_bin{int(window_size)}.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    all_pixels(window_size=7)
