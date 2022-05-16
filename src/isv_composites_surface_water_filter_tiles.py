from tqdm import tqdm
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pickle
import iris
from read_data_iris import crop_cube
import gc
import string


def land_cover_full_name(land_cover):
    full_names = {'baresparse': 'Bare/sparse vegetation',
                  'shrub': 'Shrubland',
                  'herb': 'Herbaceous vegetation',
                  'crop': 'Cropland',
                  'openforest': 'Open forest',
                  'closedforest': 'Closed forest'}
    return full_names[land_cover]


def land_cover_codes(land_cover):
    codes = {'baresparse': [60],
             'shrub': [20],
             'herb': [30],
             'shrubherb': [20, 30],
             'crop': [40],
             'openforest': [121, 122, 124, 126],
             'closedforest': [111, 112, 113, 114, 115, 116]}
    return codes[land_cover]


def land_cover_mask(land_cover, lat_south, lat_north):
    lc = iris.load('/prj/nceo/bethar/copernicus_landcover_2018.nc')[0]
    lc_tropics = crop_cube(lc, -180, 180, lat_south, lat_north)
    land_cover_mask = np.isin(lc_tropics.data, land_cover_codes(land_cover))
    return land_cover_mask


def events_for_land_cover(events, land_cover, lat_south, lat_north):
    lc_mask = land_cover_mask(land_cover, lat_south, lat_north)
    lats = np.arange(lat_south, lat_north, 0.25) + 0.5 * 0.25
    lc_events = []
    for event in events:
        if lc_mask[event[0][0], event[0][1]] and np.abs(lats[event[0][0]])<55.:
            lc_events.append(event)
    return lc_events


def time_series_around_date(data_grid, lat_idx, lon_idx, date_idx, days_range=60):
    box_whole_time_series = data_grid[:, lat_idx, lon_idx]
    end_buffer = np.ones(days_range)*np.nan
    data_pad = np.hstack((end_buffer, box_whole_time_series, end_buffer))
    time_series = data_pad[date_idx+days_range-days_range:date_idx+days_range+(days_range+1)]
    return time_series


def composite_events_all_valid(events, data_grid, imerg_anom, vod_anom, sm_anom, days_range=60, existing_composite=None, existing_n=None):
    gc.disable()
    days_around = np.arange(-days_range, days_range+1)
    if existing_composite is not None:
        composite = existing_composite
        n = existing_n
        start_idx = 0
    else:
        event = events[0]
        composite = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
        vod_series = time_series_around_date(vod_anom, event[0][0], event[0][1], event[1], days_range=days_range)
        sm_series = time_series_around_date(sm_anom, event[0][0], event[0][1], event[1], days_range=days_range)
        imerg_series = time_series_around_date(imerg_anom, event[0][0], event[0][1], event[1], days_range=days_range)
        composite[np.logical_or(np.logical_or(np.isnan(vod_series), np.isnan(sm_series)), np.isnan(imerg_series))] = np.nan
        n = (~np.isnan(composite)).astype(float)
        start_idx = 1
    for event in tqdm(events[start_idx:], desc='creating composite'):
        event_series = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
        vod_series = time_series_around_date(vod_anom, event[0][0], event[0][1], event[1], days_range=days_range)
        sm_series = time_series_around_date(sm_anom, event[0][0], event[0][1], event[1], days_range=days_range)
        imerg_series = time_series_around_date(imerg_anom, event[0][0], event[0][1], event[1], days_range=days_range)
        event_series[np.logical_or(np.logical_or(np.isnan(vod_series), np.isnan(sm_series)), np.isnan(imerg_series))] = np.nan
        additional_valid_day = np.logical_and(~np.isnan(event_series), ~np.isnan(composite))
        first_valid_day = np.logical_and(~np.isnan(event_series), np.isnan(composite))
        valid_days = np.logical_or(additional_valid_day, first_valid_day)
        n[valid_days] += 1
        composite[additional_valid_day] = composite[additional_valid_day] + (event_series[additional_valid_day] - composite[additional_valid_day])/n[additional_valid_day]
        composite[first_valid_day] = event_series[first_valid_day]
    return days_around, composite, n


def composite_events(events, data_grid, days_range=60, existing_composite=None, existing_n=None):
    gc.disable()
    days_around = np.arange(-days_range, days_range+1)
    if existing_composite is not None:
        composite = existing_composite
        n = existing_n
        start_idx = 0
    else:
        event = events[0]
        composite = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
        n = (~np.isnan(composite)).astype(float)
        start_idx = 1
    for event in tqdm(events[start_idx:], desc='creating composite'):
        event_series = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
        additional_valid_day = np.logical_and(~np.isnan(event_series), ~np.isnan(composite))
        first_valid_day = np.logical_and(~np.isnan(event_series), np.isnan(composite))
        valid_days = np.logical_or(additional_valid_day, first_valid_day)
        n[valid_days] += 1
        composite[additional_valid_day] = composite[additional_valid_day] + (event_series[additional_valid_day] - composite[additional_valid_day])/n[additional_valid_day]
        composite[first_valid_day] = event_series[first_valid_day]
    return days_around, composite, n


def hemi_composites(hem, land_covers, days_range=60):
    if hem == 'north':
        lat_south = 0
        lat_north = 60
    elif hem == 'south':
        lat_south = -60
        lat_north = 0
    else:
        raise KeyError(f'Hemisphere must be either north or south, not {hem}')
    events = pickle.load(open(f'../data/imerg_isv_events_lowpass_1std_{hem}.p','rb'))
    imerg_anomaly = iris.load(f'/scratch/bethar/daily_detrended_imerg_norm_anom_singleprecision_{hem}.nc')[0]
    imerg_anom = ma.filled(imerg_anomaly.data, np.nan).astype(np.float32)
    imerg_pad = np.ones((6940-imerg_anom.shape[0], imerg_anom.shape[1], imerg_anom.shape[2]), dtype=np.float32) * np.nan
    imerg_anom = np.concatenate((imerg_pad, imerg_anom))
    vod_anomaly = iris.load(f'/scratch/bethar/daily_detrended_vod_norm_anom_singleprecision_{hem}.nc')[0]
    vod_anom = ma.filled(vod_anomaly.data.astype(np.float32), np.nan).astype(np.float32)
    sm_anomaly = iris.load(f'/scratch/bethar/daily_detrended_sm_norm_anom_singleprecision_{hem}.nc')[0]
    sm_anom = ma.filled(sm_anomaly.data.astype(np.float32), np.nan).astype(np.float32)
    for land_cover in tqdm(land_covers):
        lc_events = events_for_land_cover(events, land_cover, lat_south, lat_north)
        days_around, imerg_composite, imerg_n = composite_events_all_valid(lc_events, imerg_anom, imerg_anom, vod_anom, sm_anom, days_range=days_range)
        _, vod_composite, vod_n = composite_events_all_valid(lc_events, vod_anom, imerg_anom, vod_anom, sm_anom, days_range=days_range)
        _, sm_composite, sm_n = composite_events_all_valid(lc_events, sm_anom, imerg_anom, vod_anom, sm_anom, days_range=days_range)
        if np.logical_and(np.allclose(imerg_n, vod_n), np.allclose(imerg_n, sm_n)):
            n = imerg_n
        else:
            raise ValueError('ns are not the same')
        np.save(f'../data/land_cover_isv_composites/imerg_composite_{hem}_55NS_{land_cover}.npy', imerg_composite)
        np.save(f'../data/land_cover_isv_composites/vod_composite_{hem}_55NS_{land_cover}.npy', vod_composite)
        np.save(f'../data/land_cover_isv_composites/sm_composite_{hem}_55NS_{land_cover}.npy', sm_composite)
        np.save(f'../data/land_cover_isv_composites/n_{hem}_55NS_{land_cover}.npy', n)


def full_composites(land_covers, days_range=60):
    # must have already run hemi_composites('south')
    events = pickle.load(open('../data/imerg_isv_events_lowpass_1std_north.p','rb'))
    imerg_anomaly = iris.load('/scratch/bethar/daily_detrended_imerg_norm_anom_singleprecision_north.nc')[0]
    imerg_anom = ma.filled(imerg_anomaly.data, np.nan).astype(np.float32)
    imerg_pad = np.ones((6940-imerg_anom.shape[0], imerg_anom.shape[1], imerg_anom.shape[2]), dtype=np.float32) * np.nan
    imerg_anom = np.concatenate((imerg_pad, imerg_anom))
    vod_anomaly = iris.load('/scratch/bethar/daily_detrended_vod_norm_anom_singleprecision_north.nc')[0]
    vod_anom = ma.filled(vod_anomaly.data.astype(np.float32), np.nan).astype(np.float32)
    sm_anomaly = iris.load('/scratch/bethar/daily_detrended_sm_norm_anom_singleprecision_north.nc')[0]
    sm_anom = ma.filled(sm_anomaly.data.astype(np.float32), np.nan).astype(np.float32)
    lat_south = 0
    lat_north = 60
    for land_cover in tqdm(land_covers):
        lc_events = events_for_land_cover(events, land_cover, lat_south, lat_north)
        imerg_composite_south = np.load(f'../data/land_cover_isv_composites/imerg_composite_south_55NS_{land_cover}.npy')
        vod_composite_south = np.load(f'../data/land_cover_isv_composites/vod_composite_south_55NS_{land_cover}.npy')
        sm_composite_south = np.load(f'../data/land_cover_isv_composites/sm_composite_south_55NS_{land_cover}.npy')
        n_south = np.load(f'../data/land_cover_isv_composites/n_south_55NS_{land_cover}.npy')
        days_around, imerg_composite, imerg_n = composite_events_all_valid(lc_events, imerg_anom, imerg_anom, vod_anom, sm_anom, 
                                                                           days_range=days_range, existing_composite=imerg_composite_south, existing_n=n_south)
        _, vod_composite, vod_n = composite_events_all_valid(lc_events, vod_anom, imerg_anom, vod_anom, sm_anom, 
                                                             days_range=days_range, existing_composite=vod_composite_south, existing_n=n_south)
        _, sm_composite, sm_n = composite_events_all_valid(lc_events, sm_anom, imerg_anom, vod_anom, sm_anom, 
                                                           days_range=days_range, existing_composite=sm_composite_south, existing_n=n_south)
        if np.logical_and(np.allclose(imerg_n, vod_n), np.allclose(imerg_n, sm_n)):
            n = imerg_n
        else:
            raise ValueError('ns are not the same')
        np.save(f'../data/land_cover_isv_composites/imerg_composite_global_55NS_{land_cover}.npy', imerg_composite)
        np.save(f'../data/land_cover_isv_composites/vod_composite_global_55NS_{land_cover}.npy', vod_composite)
        np.save(f'../data/land_cover_isv_composites/sm_composite_global_55NS_{land_cover}.npy', sm_composite)
        np.save(f'../data/land_cover_isv_composites/n_global_55NS_{land_cover}.npy', n)


def ndvi_composites(land_covers, days_range=60):
    # S Hem first
    lat_south = -60
    lat_north = 0
    events = pickle.load(open(f'../data/imerg_isv_events_lowpass_1std_south.p','rb'))
    ndvi_anomaly = iris.load(f'/scratch/bethar/monthly_detrended_ndvi_merged_norm_anom_singleprecision_south.nc')[0]
    ndvi_anom = ma.filled(ndvi_anomaly.data.astype(np.float32), np.nan).astype(np.float32)
    for land_cover in tqdm(land_covers):
        lc_events = events_for_land_cover(events, land_cover, lat_south, lat_north)
        days_around, ndvi_composite, ndvi_n = composite_events(lc_events, ndvi_anom, days_range=days_range)
        np.save(f'../data/land_cover_isv_composites/ndvi/ndvi_composite_south_55NS_{land_cover}.npy', ndvi_composite)
        np.save(f'../data/land_cover_isv_composites/ndvi/n_south_55NS_{land_cover}.npy', ndvi_n)
    # add N Hem
    lat_south = 0
    lat_north = 60
    events = pickle.load(open('../data/imerg_isv_events_lowpass_1std_north.p','rb'))
    ndvi_anomaly = iris.load(f'/scratch/bethar/monthly_detrended_ndvi_merged_norm_anom_singleprecision_north.nc')[0]
    ndvi_anom = ma.filled(ndvi_anomaly.data.astype(np.float32), np.nan).astype(np.float32)
    for land_cover in tqdm(land_covers):
        lc_events = events_for_land_cover(events, land_cover, lat_south, lat_north)
        ndvi_composite_south = np.load(f'../data/land_cover_isv_composites/ndvi/ndvi_composite_south_55NS_{land_cover}.npy')
        n_south = np.load(f'../data/land_cover_isv_composites/ndvi/n_south_55NS_{land_cover}.npy')
        days_around, ndvi_composite, ndvi_n = composite_events(lc_events, ndvi_anom,
                                                               days_range=days_range, existing_composite=ndvi_composite_south,
                                                               existing_n=n_south)
        np.save(f'../data/land_cover_isv_composites/ndvi/ndvi_composite_global_55NS_{land_cover}.npy', ndvi_composite)
        np.save(f'../data/land_cover_isv_composites/ndvi/n_global_55NS_{land_cover}.npy', ndvi_n)


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_composites(land_covers, days_range=60, ndvi=False):
    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(9, 8))
    axlist = axs.flatten()
    blue =  '#56B4E9'
    green = '#009E73'
    brown = '#E69F00'
    light_green = '#88F74D'
    alphabet = string.ascii_lowercase
    days_around = np.arange(-days_range, days_range+1)
    for i, land_cover in enumerate(land_covers):
        ax = axlist[i]
        imerg_composite = np.load(f'../data/land_cover_isv_composites/imerg_composite_global_55NS_{land_cover}.npy')
        vod_composite = np.load(f'../data/land_cover_isv_composites/vod_composite_global_55NS_{land_cover}.npy')
        sm_composite = np.load(f'../data/land_cover_isv_composites/sm_composite_global_55NS_{land_cover}.npy')
        n = np.load(f'../data/land_cover_isv_composites/n_global_55NS_{land_cover}.npy')
        ax.plot(days_around, imerg_composite, color=blue, label='precipitation')
        ax.plot(days_around, sm_composite, color=brown, label='SSM')
        ax.plot(days_around, vod_composite, color=green, label='VOD')
        if ndvi:
            ndvi_composite = np.load(f'../data/land_cover_isv_composites/ndvi/ndvi_composite_global_55NS_{land_cover}.npy')
            ax.plot(days_around, ndvi_composite, '--', color=light_green, label='NDVI')
        ax.set_ylim([-0.5, 1.75])
        ax.set_xticks(np.arange(-60, 61, 5), minor=True)
        ax.set_xticks(np.arange(-60, 61, 30))
        ax.set_yticks(np.arange(-0.5, 2., 0.5))
        ax.axhline(0, color='gray', alpha=0.3, zorder=0)
        ax.axvline(0, color='gray', alpha=0.3, zorder=0)
        ax.set_xlim([-days_range, days_range])
        ax.set_xlabel(f"days since intraseasonal\nprecipitation maximum", fontsize=14)
        ax.set_ylabel("standardised anomaly", fontsize=13)
        ax.label_outer()
        ax.tick_params(labelsize=12)
        if i == 0:
            ax.legend(loc='upper left', fontsize=11)
        ax.set_title(f'$\\bf{{({alphabet[i]})}}$ {land_cover_full_name(land_cover)}', fontsize=12)
        ax.text(58, 1.68, f'mean(n) = {int(np.round(np.mean(n)))}\nmin(n) = {int(np.min(n))}', ha='right', va='top')
    plt.tight_layout()
    save_filename = '../figures/isv_composites/no_neighbours/vod_around_precip_isv_maxima_lowpass_1std_norm_withsm_subplots_global_55NS'
    if ndvi:
        save_filename += '_ndvi_dashed'
    plt.savefig(f'{save_filename}.png', dpi=400)
    plt.savefig(f'{save_filename}.pdf', dpi=400)
    plt.savefig(f'{save_filename}.eps', dpi=400)
    plt.show()


if __name__ == '__main__':
    land_covers = ['baresparse', 'shrub', 'herb', 'crop', 'openforest', 'closedforest']
    hemi_composites('south', land_covers)
    full_composites(land_covers)
    ndvi_composites(land_covers, days_range=60)
    plot_composites(land_covers, days_range=60, ndvi=True)
