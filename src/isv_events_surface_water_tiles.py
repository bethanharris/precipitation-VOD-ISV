from tqdm import tqdm
import numpy as np
import numpy.ma as ma
import gc
import pickle
from scipy.signal import argrelextrema
import iris
from read_data_iris import read_data_all_years, daily_anomalies_normalised, daily_anomalies, monthly_anomalies_normalised
from bandpass_filters import lanczos_lowpass_filter_missing_data


def save_anomalies(hem, save_dir):
    if hem == 'north':
        lat_south = 0
        lat_north = 60
    elif hem == 'south':
        lat_south = -60
        lat_north = 0
    else:
        raise KeyError(f'Hemisphere must be either north or south, not {hem}')

    vod = read_data_all_years('VOD', band='X', min_year=2000, max_year=2018,
                              lon_west=-180, lon_east=180, lat_south=lat_south, lat_north=lat_north, mask_surface_water=True)
    imerg = read_data_all_years('IMERG', regridded=True, min_year=2000, max_year=2018,
                                lon_west=-180, lon_east=180, lat_south=lat_south, lat_north=lat_north)
    sm = read_data_all_years('SM', min_year=2000, max_year=2018,
                             lon_west=-180, lon_east=180, lat_south=lat_south, lat_north=lat_north)

    vod_anomaly = daily_anomalies_normalised(vod, detrend=True)
    vod_single = vod_anomaly.copy(data=vod_anomaly.data.astype(np.float32))
    iris.save(vod_single, f'{save_dir}/daily_detrended_vod_norm_anom_singleprecision_{hem}.nc', fill_value=-999999.0, chunksizes=(1, 240, 1440))

    del vod_anomaly
    del vod_single
    gc.collect()

    imerg_anomaly = daily_anomalies_normalised(imerg, detrend=True)
    imerg_single = imerg_anomaly.copy(data=imerg_anomaly.data.astype(np.float32))
    iris.save(imerg_single, f'{save_dir}/daily_detrended_imerg_norm_anom_singleprecision_{hem}.nc', fill_value=-999999.0, chunksizes=(1, 240, 1440))

    del imerg_anomaly
    del imerg_single
    gc.collect()

    imerg_anomaly_no_norm = daily_anomalies(imerg, detrend=True)
    imerg_single_no_norm = imerg_anomaly_no_norm.copy(data=imerg_anomaly_no_norm.data.astype(np.float32))
    iris.save(imerg_single_no_norm, f'{save_dir}/daily_detrended_imerg_anom_singleprecision_{hem}.nc', fill_value=-999999.0, chunksizes=(1, 240, 1440))

    del imerg_anomaly_no_norm
    del imerg_single_no_norm
    gc.collect()

    sm_anomaly = daily_anomalies_normalised(sm, detrend=True)
    sm_single = sm_anomaly.copy(data=sm_anomaly.data.astype(np.float32))
    iris.save(sm_single, f'{save_dir}/daily_detrended_sm_norm_anom_singleprecision_{hem}.nc', fill_value=-999999.0, chunksizes=(1, 240, 1440))

    del sm_anomaly
    del sm_single
    gc.collect()


def save_ndvi_anomalies(hem, save_dir):
    if hem == 'north':
        lat_south = 0
        lat_north = 60
    elif hem == 'south':
        lat_south = -60
        lat_north = 0
    else:
        raise KeyError(f'Hemisphere must be either north or south, not {hem}')

    aqua = read_data_all_years('NDVI', modis_sensor='aqua', min_year=2000, max_year=2018,
                               lon_west=-180, lon_east=180, lat_south=lat_south, lat_north=lat_north)
    terra = read_data_all_years('NDVI', modis_sensor='terra', min_year=2000, max_year=2018,
                                lon_west=-180, lon_east=180, lat_south=lat_south, lat_north=lat_north)    

    aqua_anomaly = monthly_anomalies_normalised(aqua, detrend=True)
    aqua_single = aqua_anomaly.copy(data=aqua_anomaly.data.astype(np.float32))
    iris.save(aqua_single, f'{save_dir}/monthly_detrended_ndvi_aqua_norm_anom_singleprecision_{hem}.nc', fill_value=-999999.0, chunksizes=(1, 240, 1440))

    del aqua_anomaly
    del aqua_single
    gc.collect()

    terra_anomaly = monthly_anomalies_normalised(terra, detrend=True)
    terra_single = terra_anomaly.copy(data=terra_anomaly.data.astype(np.float32))
    iris.save(terra_single, f'{save_dir}/monthly_detrended_ndvi_terra_norm_anom_singleprecision_{hem}.nc', fill_value=-999999.0, chunksizes=(1, 240, 1440))

    del terra_anomaly
    del terra_single
    gc.collect()


def merge_ndvi_anomalies(hem, save_dir):
    aqua_anom = iris.load(f'{save_dir}/monthly_detrended_ndvi_aqua_norm_anom_singleprecision_{hem}.nc')[0]
    terra_anom = iris.load(f'{save_dir}/monthly_detrended_ndvi_terra_norm_anom_singleprecision_{hem}.nc')[0]
    aqua_data = aqua_anom.data
    terra_data = terra_anom.data
    where_terra = ~(terra_data.mask)
    merged_data = ma.copy(aqua_data)
    merged_data[where_terra] = terra_data[where_terra]
    merged_cube = aqua_anom.copy(data=merged_data.astype(np.float32))
    iris.save(merged_cube, f'{save_dir}/monthly_detrended_ndvi_merged_norm_anom_singleprecision_{hem}.nc', fill_value=-999999.0, chunksizes=(1, 240, 1440))


def get_dates_for_box(imerg_lowfreq, lat_idx, lon_idx):
    filtered_imerg_px = imerg_lowfreq[:, lat_idx, lon_idx]
    candidate_maxima = argrelextrema(filtered_imerg_px, np.greater)[0]
    m = np.nanmean(filtered_imerg_px)
    s = np.nanstd(filtered_imerg_px)
    sig_season_maxima = filtered_imerg_px[candidate_maxima] > m + s
    sig_event_idx = candidate_maxima[sig_season_maxima]
    return sig_event_idx


def grid_coords_from_id(imerg_lowfreq, id):
    ids = np.arange(imerg_lowfreq[0].size).reshape(imerg_lowfreq.shape[1], imerg_lowfreq.shape[2])
    coords = np.where(ids==id)
    return coords[0][0], coords[1][0]


def get_all_events(imerg_lowfreq):
    events = []
    box_ids = np.arange(imerg_lowfreq[0].size)
    for box_id in tqdm(box_ids, desc='finding dates of ISV maxima'):
        lat_idx, lon_idx = grid_coords_from_id(imerg_lowfreq, box_id)
        sig_event_idx = get_dates_for_box(imerg_lowfreq, lat_idx, lon_idx)
        for event in range(len(sig_event_idx)):
            events.append(((lat_idx, lon_idx), sig_event_idx[event]))
    return events


def save_events(hem):
    imerg_anomaly = iris.load(f'/scratch/bethar/daily_detrended_imerg_anom_singleprecision_{hem}.nc')[0]
    imerg_anom = ma.filled(imerg_anomaly.data.astype(np.float32), np.nan)
    imerg_pad = np.ones((6940-imerg_anom.shape[0], imerg_anom.shape[1], imerg_anom.shape[2]), dtype=np.float32) * np.nan
    imerg_anom = np.concatenate((imerg_pad, imerg_anom))

    imerg_grid_size = imerg_anom[0].shape

    imerg_lowfreq = np.empty_like(imerg_anom, dtype=np.float32)
    for i in tqdm(range(imerg_anom.shape[1]), desc='filtering IMERG to ISV'):
        for j in range(imerg_anom.shape[2]):
            imerg_lowfreq[:, i, j] = lanczos_lowpass_filter_missing_data(imerg_anom[:, i, j], 1./25., 
                                                                         window=121, min_slice_size=100)

    events = get_all_events(imerg_lowfreq)
    pickle.dump(events, open(f'../data/imerg_isv_events_lowpass_1std_{hem}.p', 'wb'))


if __name__ == '__main__':
    print('save standardised anomalies N Hemisphere')
    save_anomalies('north')
    save_ndvi_anomalies('north')
    merge_ndvi_anomalies('north')
    print('save standardised anomalies S Hemisphere')
    save_anomalies('south')
    save_ndvi_anomalies('south')
    merge_ndvi_anomalies('south')
    print('find ISV events N Hemisphere')
    save_events('north')
    print('find ISV events S Hemisphere')
    save_events('south')
