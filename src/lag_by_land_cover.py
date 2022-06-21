import numpy as np
import re
import iris
from read_data_iris import crop_cube
import scipy.stats
from read_csagan_saved_output import read_region_data
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
from plot_utils import binned_cmap, StripyPatch
import cartopy.feature as feat
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from lag_subplots import tile_global_validity, tile_global_from_saved_spectra


lats_south = {'global': -55, 'southern': -60, 'tropics': -30, 'northern': 30, 'polar': 60}
lats_north = {'global': 55, 'southern': -30, 'tropics': 30, 'northern': 60, 'polar': 80}


def copernicus_land_cover(lon_west=-180, lon_east=180, lat_south=-60, lat_north=60):
    lc = iris.load('/prj/nceo/bethar/copernicus_landcover_2018.nc')[0]
    lc_region = crop_cube(lc, lon_west, lon_east, lat_south, lat_north)
    coded_land_cover_array = lc_region.data
    string_array = np.empty((coded_land_cover_array.shape[0], coded_land_cover_array.shape[1]), dtype='U22')
    code_dict = {
    0: 'Missing',
    20: 'Shrubland',
    30: 'Herbaceous vegetation',
    40: 'Cropland',
    50: 'Urban',
    60: 'Bare/sparse vegetation',
    70: 'Snow/ice',
    80: 'Inland water',
    90: 'Herbaceous wetland',
    100: 'Moss/lichen',
    111: 'Closed forest',
    112: 'Closed forest',
    113: 'Closed forest',
    114: 'Closed forest',
    115: 'Closed forest',
    116: 'Closed forest',
    121: 'Open forest',
    122: 'Open forest',
    123: 'Open forest',
    124: 'Open forest',
    125: 'Open forest',
    126: 'Open forest',
    200: 'Open water'
    }
    for i in range(coded_land_cover_array.shape[0]):
        for j in range(coded_land_cover_array.shape[1]):
            code = coded_land_cover_array[i, j]
            string_array[i, j] = code_dict[code]
    return string_array


def line_break_string(the_string, max_line_length):
    if len(the_string) <= max_line_length:
        return the_string
    else:
        spaces = re.finditer(' ', the_string)
        space_positions = np.array([space.start() for space in spaces])
        if np.any(space_positions < max_line_length):
            space_position = space_positions[space_positions < max_line_length][-1]
        else:
            space_position = space_positions[0]
        broken_string = the_string[0:space_position] + '\n' + the_string[space_position+1:]
        return broken_string


def all_season_lags(spectra_save_dir, tile, band_days_lower, band_days_upper, nonzero_only=False):
    seasons = ['MAM', 'JJA', 'SON', 'DJF']
    mask_lag = {}
    for season in seasons:
        if tile != 'global':
            tile_filename = f"{spectra_save_dir}/spectra_nooverlap_{tile}_IMERG_VOD_X_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.p"
            neighbourhood_average_spectra = pickle.load(open(tile_filename,'rb'))
            for key in neighbourhood_average_spectra.keys():
                neighbourhood_average_spectra[key] = neighbourhood_average_spectra[key][20:-20]
        else:
            neighbourhood_average_spectra = tile_global_from_saved_spectra(spectra_save_dir, season, band_days_lower, band_days_upper)
        if nonzero_only:
            mask_lag_mean = neighbourhood_average_spectra['lag']
            mask_lag_error = neighbourhood_average_spectra['lag_error']
            lags_nonzero_only = np.copy(mask_lag_mean)
            lags_upper = lags_nonzero_only + mask_lag_error
            lags_lower = lags_nonzero_only - mask_lag_error
            confidence_interval_overlaps_zero = (np.sign(lags_lower)/np.sign(lags_upper) == -1)
            lags_nonzero_only[confidence_interval_overlaps_zero] = np.nan
            mask_lag[season] = lags_nonzero_only
        else:
            mask_lag[season] = neighbourhood_average_spectra['lag']
    return mask_lag


def median_95ci_width(spectra_save_dir, tile, band_days_lower, band_days_upper):
    seasons = ['MAM', 'JJA', 'SON', 'DJF']
    all_lag_errors = {}
    for season in seasons:
        if tile != 'global':
            tile_filename = f"{spectra_save_dir}/spectra_nooverlap_{tile}_IMERG_VOD_X_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.p"
            neighbourhood_average_spectra = pickle.load(open(tile_filename,'rb'))
            for key in neighbourhood_average_spectra.keys():
                if tile == 'polar':
                    neighbourhood_average_spectra[key] = neighbourhood_average_spectra[key][20:-40]
                else:
                    neighbourhood_average_spectra[key] = neighbourhood_average_spectra[key][20:-20]
        else:
            neighbourhood_average_spectra = tile_global_from_saved_spectra(spectra_save_dir, season, band_days_lower, band_days_upper)
        all_lag_errors[season] = neighbourhood_average_spectra['lag_error']
    lag_errors_stack = np.stack([v for v in all_lag_errors.values()])
    return np.nanpercentile(lag_errors_stack, 50)


def all_season_validity(spectra_save_dir, tile, band_days_lower, band_days_upper):
    seasons = ['MAM', 'JJA', 'SON', 'DJF']
    validity = {}
    for season in seasons:
        if tile != 'global':
            tile_filename = f"{spectra_save_dir}/spectra_nooverlap_tropics_IMERG_VOD_X_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.p"
            neighbourhood_average_spectra = pickle.load(open(tile_filename,'rb'))
            original_output_filename = f"{spectra_save_dir}/tropics_IMERG_VOD_spectra_X_{season}_mask_sw_best85.p"
            lats, lons, spectra = read_region_data(original_output_filename, tile, -180, 180, lats_south[tile], lats_north[tile])
            no_csa = (spectra == {})
            for key in neighbourhood_average_spectra.keys():
                if tile == 'polar':
                    neighbourhood_average_spectra[key] = neighbourhood_average_spectra[key][20:-40]
                else:
                    neighbourhood_average_spectra[key] = neighbourhood_average_spectra[key][20:-20]
        else:
            neighbourhood_average_spectra = tile_global_from_saved_spectra(spectra_save_dir, season, band_days_lower, band_days_upper)
            no_csa = tile_global_validity(spectra_save_dir, season)
        coherencies = neighbourhood_average_spectra['coherency']
        validity[season] = (coherencies >= 0.7795).astype(int)
        validity[season][no_csa] = 2
    return validity


def hist_lc(mask_lag, lc_codes, land_cover_code, density=False):
    all_lags = []
    for season in mask_lag.keys():
        if land_cover_code == 'all':
            all_lc_codes = ['Bare/sparse vegetation','Herbaceous vegetation',
                            'Shrubland','Cropland','Open forest', 'Closed forest']
            season_lags = mask_lag[season][np.isin(lc_codes, all_lc_codes)]
        else:
            season_lags = mask_lag[season][lc_codes==land_cover_code]
        all_lags += season_lags.tolist()
    hist, _ = np.histogram(all_lags, bins=np.arange(-30, 31, 1), density=density)
    return hist


def subplots(spectra_save_dir, density=False, show_95ci=True, all_lc_line=False):
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[1, 0:2])
    ax3 = fig.add_subplot(gs[1, 2:])
    plt.subplots_adjust(hspace=0.45, wspace=0.4)
    lc_codes = copernicus_land_cover(lat_south=-55, lat_north=55)
    lc_array = np.zeros_like(lc_codes, dtype=int)
    lc_array[lc_codes=='Bare/sparse vegetation'] = 1
    lc_array[lc_codes=='Herbaceous vegetation'] = 2
    lc_array[lc_codes=='Shrubland'] = 3
    lc_array[lc_codes=='Cropland'] = 4
    lc_array[lc_codes=='Open forest'] = 5
    lc_array[lc_codes=='Closed forest'] = 6
    lons = np.arange(-180, 180, 0.25) + 0.125
    lats = np.arange(-55, 55, 0.25) + 0.125
    lon_bounds = np.hstack((lons - 0.5*0.25, np.array([lons[-1]+0.5*0.25])))
    lat_bounds = np.hstack((lats - 0.5*0.25, np.array([lats[-1]+0.5*0.25])))
    levels = np.arange(1, 8) - 0.5
    lc_colors = ['#ffd92f', '#8da0cb', '#fc8d62', '#e78ac3','#a6d854', '#66c2a5']
    lc_cmap, lc_norm = binned_cmap(levels, 'tab10', fix_colours=[(i, c) for i, c in enumerate(lc_colors)])
    lc_cmap.set_bad('w')
    lc_cmap.set_under('w')
    p = ax1.pcolormesh(lon_bounds, lat_bounds, lc_array, transform=ccrs.PlateCarree(), 
                       cmap=lc_cmap, norm=lc_norm, rasterized=True)
    fig_left = ax2.get_position().x0
    fig_right = ax3.get_position().x1
    cax = fig.add_axes([fig_left, ax1.get_position().y0-0.075, fig_right-fig_left, 0.03])
    cbar = fig.colorbar(p, orientation='horizontal', cax=cax, aspect=40, pad=0.12)
    cbar.set_ticks(levels[:-1] + 0.5)
    cbar.ax.set_xticklabels(['Bare/sparse \n vegetation', 'Herbaceous \n vegetation', 'Shrubland', 'Cropland', 'Open forest', 'Closed forest'])
    cbar.ax.tick_params(labelsize=14)
    ax1.set_extent((-180, 180, -55, 55), crs=ccrs.PlateCarree())
    ax1.coastlines(color='black', linewidth=1)
    ax1.set_xticks(np.arange(-90, 91, 90), crs=ccrs.PlateCarree())
    ax1.set_yticks(np.arange(-50, 51, 50), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.tick_params(labelsize=14)
    ax1.tick_params(axis='x', pad=5)
    ax1.set_title("$\\bf{(a)}$" + ' Modal land cover class (Copernicus 2018)', fontsize=14)    
    lag_2540 = all_season_lags(spectra_save_dir, 'global', 25, 40)
    lc_codes = copernicus_land_cover(lat_south=-55, lat_north=55)
    lc_colors = {'Bare/sparse vegetation': '#ffd92f',
                 'Herbaceous vegetation': '#8da0cb',
                 'Shrubland': '#fc8d62',
                 'Cropland': '#e78ac3',
                 'Open forest': '#a6d854',
                 'Closed forest': '#66c2a5'}
    lag_bins = np.arange(-30, 31, 1)
    bin_centres = lag_bins[:-1] + (lag_bins[1] - lag_bins[0])/2.
    for land_cover_code in lc_colors.keys():
        hist = hist_lc(lag_2540, lc_codes, land_cover_code, density=density)
        ax2.plot(bin_centres, hist, '-o', color=lc_colors[land_cover_code], ms=3)
    if all_lc_line:
        hist = hist_lc(lag_2540, lc_codes, 'all', density=density)
        ax2.plot(bin_centres, hist, '--', color='k', ms=0, linewidth=1)
    ax2.tick_params(labelsize=14)
    ax2.set_xlim([-30, 30])
    ax2.set_xlabel('phase difference (days)', fontsize=16)
    if density:
        ax2.set_ylabel('pdf', fontsize=16)
    else:
        ax2.set_ylabel('number of pixels', fontsize=16)
    if show_95ci:
        median_lag_error = median_95ci_width(spectra_save_dir, 'global', 25, 40)
        ylims = ax2.get_ylim()
        y_range = ylims[1]-ylims[0]
        ci_line = Rectangle((27.-2*median_lag_error, ylims[0]+0.925*y_range), 2*median_lag_error, 0.0025*y_range, edgecolor='k', facecolor='k')
        line_middle = [27.-median_lag_error, ylims[0]+0.92625*y_range]
        ax2.add_artist(ci_line)
        ax2.plot(line_middle[0], line_middle[1], 'k-o', ms=3)
        ax2.text(line_middle[0], line_middle[1]-0.075*y_range, '95% CI', fontsize=12, transform=ax2.transData, ha='center')
    ax2.text(0.03, 0.9, '$\\bf{(b)}$', fontsize=14, transform=ax2.transAxes)
    ax2.text(0.03, 0.82, u'25\u201340 days', fontsize=14, transform=ax2.transAxes)
    ax2.axvline(0, color='gray', alpha=0.3, zorder=0)
    lag_4060 = all_season_lags(spectra_save_dir, 'global', 40, 60)
    for land_cover_code in lc_colors.keys():
        hist = hist_lc(lag_4060, lc_codes, land_cover_code, density=density)
        ax3.plot(bin_centres, hist, '-o', color=lc_colors[land_cover_code], ms=3)
    if all_lc_line:
        hist = hist_lc(lag_4060, lc_codes, 'all', density=density)
        ax3.plot(bin_centres, hist, '--', color='k', ms=0, linewidth=1)
    ax3.tick_params(labelsize=14)
    ax3.set_xlim([-30, 30])
    ax3.set_xlabel('phase difference (days)', fontsize=16)
    if show_95ci:
        median_lag_error = median_95ci_width(spectra_save_dir, 'global', 40, 60)
        ylims = ax3.get_ylim()
        y_range = ylims[1]-ylims[0]
        ci_line = Rectangle((27.-2*median_lag_error, ylims[0]+0.925*y_range), 2*median_lag_error, 0.0025*y_range, edgecolor='k', facecolor='k')
        line_middle = [27.-median_lag_error, ylims[0]+0.92625*y_range]
        ax3.add_artist(ci_line)
        ax3.plot(line_middle[0], line_middle[1], 'k-o', ms=3)
        ax3.text(line_middle[0], line_middle[1]-0.075*y_range, '95% CI', fontsize=12, transform=ax3.transData, ha='center')
    ax3.text(0.03, 0.9, '$\\bf{(c)}$', fontsize=14, transform=ax3.transAxes)
    ax3.text(0.03, 0.82, u'40\u201360 days', fontsize=14, transform=ax3.transAxes)
    ax3.axvline(0, color='gray', alpha=0.3, zorder=0)
    save_filename = '../figures/land_cover_subplots_global'
    if density:
        save_filename += '_density'
    if show_95ci:
        save_filename += '_median95ci'
    if all_lc_line:
        save_filename += '_showall'
    plt.savefig(f'{save_filename}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{save_filename}.pdf', dpi=900, bbox_inches='tight')
    plt.show()


def plot_global_percent_validity(spectra_save_dir, ax, band_days_lower, band_days_upper):
    land_cover_codes = ['Bare/sparse vegetation','Herbaceous vegetation',
     'Shrubland','Cropland','Open forest', 'Closed forest']
    tile = 'global'
    validity = all_season_validity(spectra_save_dir, tile, band_days_lower, band_days_upper)
    seasons = validity.keys()
    tile_lat_south = lats_south[tile]
    tile_lat_north = lats_north[tile]
    lc_codes = copernicus_land_cover(lat_south=tile_lat_south, lat_north=tile_lat_north)
    coherent_pixels = {code: 0 for code in land_cover_codes}
    incoherent_pixels = {code: 0 for code in land_cover_codes}
    no_obs_pixels = {code: 0 for code in land_cover_codes}
    total_pixels = {code: 0 for code in land_cover_codes}
    for code in land_cover_codes:
        for season in seasons:
            coherent_pixels[code] += np.logical_and(validity[season]==1, lc_codes==code).sum()
            incoherent_pixels[code] += np.logical_and(validity[season]==0, lc_codes==code).sum()
            no_obs_pixels[code] += np.logical_and(validity[season]==2, lc_codes==code).sum()
            total_pixels[code] += (lc_codes==code).sum()
    coherent_list = np.array([value for value in coherent_pixels.values()])
    incoherent_list = np.array([value for value in incoherent_pixels.values()])
    no_obs_list = np.array([value for value in no_obs_pixels.values()])
    total_list = np.array([value for value in total_pixels.values()])
    coherent_percent = 100.*coherent_list/total_list
    incoherent_percent = 100.*incoherent_list/total_list
    no_obs_percent = 100.*no_obs_list/total_list
    coherent_percent_of_obs = 100.*(coherent_list/(coherent_list+incoherent_list))
    land_cover_labels = [line_break_string(label, 10) for label in land_cover_codes]
    width = 1
    ax.bar(land_cover_labels, coherent_percent, width, color=['#ffd92f','#8da0cb','#fc8d62','#e78ac3','#a6d854','#66c2a5'], edgecolor='k', linewidth=0.75, label='Coherency\n> 95% CL')
    ax.bar(land_cover_labels, incoherent_percent, width, color='#ffffff', edgecolor='k', linewidth=0.75, bottom=coherent_percent, label='No coherency\n> 95% CL')
    ax.bar(land_cover_labels, no_obs_percent, width, color='#cccccc', edgecolor='k', linewidth=0.75, bottom=coherent_percent+incoherent_percent, label='Insufficient obs')
    ax.set_ylim([0, 100])
    ax.set_xlim([-0.5, len(land_cover_codes)-0.5])
    ax.tick_params(labelsize=14)
    ax.tick_params(axis='x', labelrotation=90)
    for i in range(total_list.size):
        ax.text(i, 102, f'({total_list[i]})', fontsize=10, horizontalalignment='center')


def subplots_percent_validity(spectra_save_dir):
    fig, (ax2540, ax4060) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_global_percent_validity(spectra_save_dir, ax2540, 25, 40)
    plot_global_percent_validity(spectra_save_dir, ax4060, 40, 60)
    plt.subplots_adjust(right=0.8, wspace=0.05)
    color_list = ['#ffd92f','#8da0cb','#fc8d62','#e78ac3','#a6d854','#66c2a5']
    greys = ['#cccccc']*6
    whites = ['#ffffff']*6
    cmaps = [color_list, whites, greys]
    cmap_labels = ['Coherency\n> 95% CL', 'No coherency\n> 95% CL', 'Insufficient obs']
    cmap_handles = [Rectangle((0, 0), 1, 1, edgecolor='k', linewidth=0.75) for _ in cmaps]
    handler_map = dict(zip(cmap_handles, 
                           [StripyPatch(cm) for cm in cmaps]))
    bar_legend = ax4060.legend(handles=cmap_handles, labels=cmap_labels, handler_map=handler_map, fontsize=12,
                               loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=True)
    # this bit just uses a second legend to put nice borders around the legend patches
    # and hide artefacts in the non-striped boxes
    dummy_labels = ['\n', '\n', '']
    dummy_colours = ['none', '#ffffff', '#cccccc']
    border_boxes = [Rectangle((0, 0), 1.05, 1.05, fc=fc, edgecolor='k', linewidth=0.5) for fc in dummy_colours]
    legend_borders = ax4060.legend(handles=border_boxes, labels=dummy_labels,
                                   fontsize=12, loc='center left', bbox_to_anchor=(1.05, 0.5), 
                                   frameon=False)
    ax4060.add_artist(bar_legend)
    ax2540.set_ylabel(r'% of pixels', fontsize=16)
    ax2540.set_title("$\\bf{(a)}$" + u" 25\u201340 days", fontsize=16, pad=25)
    ax4060.set_title("$\\bf{(b)}$" + u" 40\u201360 days", fontsize=16, pad=25)
    save_filename = f'../figures/validity_percentage_by_land_cover_global_subplots'
    plt.savefig(f'{save_filename}.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'{save_filename}.png', dpi=600, bbox_inches='tight')
    plt.show()


def median_values(spectra_save_dir, band_days_lower, band_days_upper):
    lc_codes = copernicus_land_cover(lat_south=-55, lat_north=55)
    lc_array = np.zeros_like(lc_codes, dtype=int)
    lc_array[lc_codes=='Bare/sparse vegetation'] = 1
    lc_array[lc_codes=='Herbaceous vegetation'] = 2
    lc_array[lc_codes=='Shrubland'] = 3
    lc_array[lc_codes=='Cropland'] = 4
    lc_array[lc_codes=='Open forest'] = 5
    lc_array[lc_codes=='Closed forest'] = 6
    lag_band = all_season_lags(spectra_save_dir, 'global', band_days_lower, band_days_upper)
    lc_code_list = ['Bare/sparse vegetation', 'Herbaceous vegetation', 'Shrubland',
                 'Cropland', 'Open forest', 'Closed forest']
    for land_cover_code in lc_code_list:
        all_lags = []
        for season in lag_band.keys():
            season_lags = lag_band[season][lc_codes==land_cover_code]
            all_lags += season_lags.tolist()
        valid_lags = np.array(all_lags)
        valid_lags = valid_lags[~np.isnan(valid_lags)]
        print(f'{land_cover_code}: median {np.median(valid_lags): 0.2f} days, mode: {scipy.stats.mode(np.round(valid_lags)).mode[0]} days')


if __name__ == '__main__':
    spectra_save_dir = '/prj/nceo/bethar/cross_spectral_analysis_results/test/'
    subplots(spectra_save_dir, density=True, show_95ci=True, all_lc_line=True)
    # subplots_percent_validity(spectra_save_dir)
 