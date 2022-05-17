import pickle
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from tqdm import tqdm
import os
from read_csagan_saved_output import read_region_data


def tile_global_from_saved_spectra(spectra_save_dir, season, band_days_lower, band_days_upper):
    tropics_filename = f"{spectra_save_dir}/spectra_nooverlap_tropics_IMERG_VOD_X_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.p"
    northern_filename = f"{spectra_save_dir}/spectra_nooverlap_northern_IMERG_VOD_X_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.p"
    southern_filename = f"{spectra_save_dir}/spectra_nooverlap_southern_IMERG_VOD_X_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.p"
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


def tile_global_validity(spectra_save_dir, season):
    tropics_filename = f"{spectra_save_dir}/tropics_IMERG_VOD_spectra_X_{season}_mask_sw_best85.p"
    northern_filename = f"{spectra_save_dir}/northern_IMERG_VOD_spectra_X_{season}_mask_sw_best85.p"
    southern_filename = f"{spectra_save_dir}/southern_IMERG_VOD_spectra_X_{season}_mask_sw_best85.p"
    _, _, spectra_tropics = read_region_data(tropics_filename, 'tropics', -180, 180, -30, 30)
    _, _, spectra_northern = read_region_data(northern_filename, 'northern', -180, 180, 30, 55)
    _, _, spectra_southern = read_region_data(southern_filename, 'southern', -180, 180, -55, -30)
    no_csa_global = np.zeros((440, 1440), dtype='bool')
    no_csa_global[0:100] = (spectra_southern == {})
    no_csa_global[100:340] = (spectra_tropics == {})
    no_csa_global[340:] = (spectra_northern == {})
    return no_csa_global


def save_lags_to_file(spectra_save_dir):
    os.system('mkdir -p ../data/lag_subplots_data')
    seasons = np.repeat(['MAM', 'JJA', 'SON', 'DJF'], 2)
    band_days_lower = [25, 40]*4
    band_days_upper = [40, 60]*4
    for i in tqdm(range(seasons.size), desc='saving lag data to file'):
        season = seasons[i]
        lower = band_days_lower[i]
        upper = band_days_upper[i]
        lag_dict = tile_global_from_saved_spectra(spectra_save_dir, season, lower, upper)
        lag = lag_dict['lag']
        period = lag_dict['period']
        lag_error = lag_dict['lag_error']
        no_csa = tile_global_validity(spectra_save_dir, season)
        np.save(f'../data/lag_subplots_data/lag_{season}_{lower}-{upper}.npy', lag)
        np.save(f'../data/lag_subplots_data/lag_error_{season}_{lower}-{upper}.npy', lag_error)
        np.save(f'../data/lag_subplots_data/period_{season}_{lower}-{upper}.npy', period)
        np.save(f'../data/lag_subplots_data/no_csa_{season}_{lower}-{upper}.npy', no_csa)


def global_plots_mean_estimate():
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    lons = np.arange(-180, 180, 0.25) + 0.5*0.25
    lats = np.arange(-55, 55, 0.25) + 0.5*0.25
    lon_bounds = np.hstack((lons - 0.5*0.25, np.array([lons[-1]+0.5*0.25])))
    lat_bounds = np.hstack((lats - 0.5*0.25, np.array([lats[-1]+0.5*0.25])))
    fig = plt.figure(figsize=(16, 10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(4, 2),
                    axes_pad=0.2,
                    cbar_location='bottom',
                    cbar_mode='single',
                    cbar_pad=0.15,
                    cbar_size='10%',
                    label_mode='')  # note the empty label_mode

    seasons = np.repeat(['MAM', 'JJA', 'SON', 'DJF'], 2)
    band_days_lower = [25, 40]*4
    band_days_upper = [40, 60]*4

    contour_multiple = 5.
    lag_contour_levels = np.arange(-30., 31., contour_multiple).astype(int)
    colormap = cm.get_cmap('RdYlBu_r', 2*(lag_contour_levels.size+7))
    colors = list(colormap(np.arange(2*(lag_contour_levels.size+7))))
    colors_to_take = [0, 1, 2, 4, 7, 12, 27, 32,35, 37, 38, 39] 
    no_centre_colors = [colors[i] for i in colors_to_take]

    cmap = mpl.colors.ListedColormap(no_centre_colors, "")
    norm = mpl.colors.BoundaryNorm(lag_contour_levels, ncolors=len(lag_contour_levels)-1, clip=False)
    cmap.set_bad('#e7e7e7')
    cmap.set_under('white')

    for i, ax in enumerate(axgr):
        lag = np.load(f'../data/lag_subplots_data/lag_{seasons[i]}_{band_days_lower[i]}-{band_days_upper[i]}.npy')
        lag_for_hist = np.copy(lag)
        total_lags = (~np.isnan(lag_for_hist)).sum()
        neg_lags = (lag_for_hist < 0.).sum()
        pos_lags = (lag_for_hist > 0.).sum()
        percent_neg = np.round(float(neg_lags)/float(total_lags) * 100.)
        percent_pos = np.round(float(pos_lags)/float(total_lags) * 100.)
        no_csa = np.load(f'../data/lag_subplots_data/no_csa_{seasons[i]}_{band_days_lower[i]}-{band_days_upper[i]}.npy')
        invalid_but_csa = np.logical_and(~no_csa, np.isnan(lag))
        lag[invalid_but_csa] = -999
        ax.coastlines(color='#999999',linewidth=0.1)
        ax.text(0.015, 0.825, f'{seasons[i]}', fontsize=16, transform=ax.transAxes)
        p = ax.pcolormesh(lon_bounds, lat_bounds, lag, transform=ccrs.PlateCarree(), 
                          cmap=cmap, norm=norm, rasterized=True)
        ax.set_extent((-180, 180, -55, 55), crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(-90, 91, 90), crs=projection)
        ax.set_yticks(np.arange(-50, 51, 50), crs=projection)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize=14)
        ax.tick_params(axis='x', pad=5)
        ax_sub = inset_axes(ax, width=1,
                       height=0.7, loc='lower left',
                       bbox_to_anchor=(-175, -50),
                       bbox_transform=ax.transData,
                       borderpad=0)
        N, bins, patches = ax_sub.hist(lag_for_hist.ravel(), bins=lag_contour_levels)
        for i in range(len(patches)):
            patches[i].set_facecolor(no_centre_colors[i])
        ax_sub.set_yticks([])
        ax_sub.set_xticks([])
        ax_sub.patch.set_alpha(0.)
        ax_sub.spines['right'].set_visible(False)
        ax_sub.spines['left'].set_visible(False)
        ax_sub.spines['top'].set_visible(False)
        # add detail on percent pos/neg
        ymin, ymax = ax_sub.get_ylim()
        ax_sub.set_ylim(top=1.3 * ymax)
        ax_sub.axvline(0, color='k', linestyle='--', linewidth=0.5, dashes=(5, 5))
        ax_sub.text(0.55, 0.82, f'{int(percent_pos):d}%', transform=ax_sub.transAxes, horizontalalignment='left', fontsize=10)
        ax_sub.text(0.45, 0.82, f'{int(percent_neg):d}%', transform=ax_sub.transAxes, horizontalalignment='right', fontsize=10)
    axes = np.reshape(axgr, axgr.get_geometry())
    for ax in axes[:-1, :].flatten():
        ax.xaxis.set_tick_params(which='both', 
                                 labelbottom=False, labeltop=False)
    for ax in axes[:, 1:].flatten():
        ax.yaxis.set_tick_params(which='both', 
                                 labelbottom=False, labeltop=False)
    axes = np.reshape(axgr, axgr.get_geometry())
    axes[0, 0].set_title(u"25\u201340 days", fontsize=18)
    axes[0, 1].set_title(u"40\u201360 days", fontsize=18)    
    cbar = axgr.cbar_axes[0].colorbar(p)
    cbar.ax.tick_params(labelsize=16) 
    cbar.ax.set_xlabel('phase difference (days)', fontsize=18)
    plt.savefig('../figures/lag_subplots_mean_phase_diff_estimate.png', dpi=600, bbox_inches='tight')
    plt.savefig('../figures/lag_subplots_mean_phase_diff_estimate.pdf', dpi=1000, bbox_inches='tight')
    plt.show()


def global_plots_with95ci():
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    lons = np.arange(-180, 180, 0.25) + 0.5*0.25
    lats = np.arange(-55, 55, 0.25) + 0.5*0.25
    lon_bounds = np.hstack((lons - 0.5*0.25, np.array([lons[-1]+0.5*0.25])))
    lat_bounds = np.hstack((lats - 0.5*0.25, np.array([lats[-1]+0.5*0.25])))
    fig = plt.figure(figsize=(16, 10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(4, 2),
                    axes_pad=0.2,
                    cbar_location='bottom',
                    cbar_mode='single',
                    cbar_pad=0.15,
                    cbar_size='10%',
                    label_mode='')
    seasons = np.repeat(['MAM', 'JJA', 'SON', 'DJF'], 2)
    band_days_lower = [25, 40]*4
    band_days_upper = [40, 60]*4
    positive_colour = '#B0154B'
    unsure_colour = '#E1BE6A'
    negative_colour = '#6072C1'
    lag_colours = [negative_colour, unsure_colour, positive_colour]
    cmap = mpl.colors.ListedColormap(lag_colours, "")
    norm = mpl.colors.BoundaryNorm(np.arange(4)-0.5, ncolors=3, clip=False)
    cmap.set_bad('white')
    cmap.set_under('#c7c7c7')
    for i, ax in enumerate(axgr):
        lag = np.load(f'../data/lag_subplots_data/lag_{seasons[i]}_{band_days_lower[i]}-{band_days_upper[i]}.npy')
        lag_sign = np.ones_like(lag) * np.nan
        lag_error = np.load(f'../data/lag_subplots_data/lag_error_{seasons[i]}_{band_days_lower[i]}-{band_days_upper[i]}.npy')
        lag_upper = lag + lag_error
        lag_lower = lag - lag_error
        positive_confidence_interval = (lag_lower > 0.)
        lag_sign[positive_confidence_interval] = 2
        negative_confidence_interval = (lag_upper < 0.)
        lag_sign[negative_confidence_interval] = 0
        confidence_interval_overlaps_zero = (np.sign(lag_upper)/np.sign(lag_lower) == -1)
        lag_sign[confidence_interval_overlaps_zero] = 1
        total_lags = (~np.isnan(lag)).sum()
        percent_neg = np.round(float(negative_confidence_interval.sum())/float(total_lags) * 100.)
        percent_pos = np.round(float(positive_confidence_interval.sum())/float(total_lags) * 100.)
        percent_unsure = np.round(float(confidence_interval_overlaps_zero.sum())/float(total_lags) * 100.)
        no_csa = np.load(f'../data/lag_subplots_data/no_csa_{seasons[i]}_{band_days_lower[i]}-{band_days_upper[i]}.npy')
        invalid_but_csa = np.logical_and(~no_csa, np.isnan(lag))
        lag_sign[invalid_but_csa] = -999
        ax.coastlines(color='#999999',linewidth=0.1)
        ax.text(0.015, 0.825, f'{seasons[i]}', fontsize=16, transform=ax.transAxes)
        p = ax.pcolormesh(lon_bounds, lat_bounds, lag_sign, transform=ccrs.PlateCarree(), 
                          cmap=cmap, norm=norm, rasterized=True)
        ax.set_extent((-180, 180, -55, 55), crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(-90, 91, 90), crs=projection)
        ax.set_yticks(np.arange(-50, 51, 50), crs=projection)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize=14)
        ax.tick_params(axis='x', pad=5)
        ax.text(0.05, 0.05, f'{int(percent_neg):d}%', color=negative_colour, transform=ax.transAxes, horizontalalignment='center', fontsize=12)
        ax.text(0.13, 0.05, f'{int(percent_unsure):d}%', color='#ba8e25', transform=ax.transAxes, horizontalalignment='center', fontsize=12)
        ax.text(0.21, 0.05, f'{int(percent_pos):d}%', color=positive_colour, transform=ax.transAxes, horizontalalignment='center', fontsize=12)
    axes = np.reshape(axgr, axgr.get_geometry())
    for ax in axes[:-1, :].flatten():
        ax.xaxis.set_tick_params(which='both', 
                                 labelbottom=False, labeltop=False)
    for ax in axes[:, 1:].flatten():
        ax.yaxis.set_tick_params(which='both', 
                                 labelbottom=False, labeltop=False)
    axes = np.reshape(axgr, axgr.get_geometry())
    axes[0, 0].set_title(u"25\u201340 days", fontsize=18)
    axes[0, 1].set_title(u"40\u201360 days", fontsize=18)    
    cbar = axgr.cbar_axes[0].colorbar(p, ticks=[0, 1, 2])
    cbar.ax.set_xticklabels(['negative\nphase difference', 
                             'phase difference\nindistinguishable from zero',
                             'positive\nphase difference'])
    cbar.ax.tick_params(labelsize=16)
    os.system('mkdir -p ../figures')
    plt.savefig('../figures/lag_subplots_with95ci.png', dpi=600, bbox_inches='tight')
    plt.savefig('../figures/lag_subplots_with95ci.pdf', dpi=1000, bbox_inches='tight')
    plt.show()


def lag_sign_stats(season, band_days_lower, band_days_upper):
    lag = np.load(f'../data/lag_subplots_data/lag_{season}_{band_days_lower}-{band_days_upper}.npy')
    lag_sign = np.ones_like(lag) * np.nan
    lag_error = np.load(f'../data/lag_subplots_data/lag_error_{season}_{band_days_lower}-{band_days_upper}.npy')
    lag_lower = lag - lag_error
    lag_upper = lag + lag_error
    total_px = (~np.isnan(lag)).sum()
    pos_px = (lag > 0.).sum()
    neg_px = (lag < 0.).sum()
    pos_less_7 = np.logical_and(lag>0., lag<7.).sum()
    pos_less_10 = np.logical_and(lag>0., lag<10.).sum()
    pos_ci_px = (lag_lower>0.).sum()
    neg_ci_px = (lag_upper<0.).sum()
    cross_ci = np.logical_and(lag_upper>0., lag_lower<0.)
    cross_ci_px = (cross_ci).sum()
    print(f'total pixels: {total_px}')
    print(f'positive lag: {pos_px}')
    print(f'negative lag: {neg_px}')
    print(f'positive and less than 7: {pos_less_7}')
    print(f'positive and less than 10: {pos_less_10}')
    print('Accounting for 95% CI:')
    print(f'positive: {pos_ci_px}')
    print(f'negative: {neg_ci_px}')
    print(f'sign uncertain: {cross_ci_px}')


if __name__ == '__main__':
    spectra_save_dir = '/prj/nceo/bethar/cross_spectral_analysis_results/test/'
    save_lags_to_file(spectra_save_dir)
    global_plots_with95ci()
