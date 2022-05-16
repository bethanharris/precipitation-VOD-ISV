import os
import time
import numpy as np
import numpy.ma as ma
import xarray as xr
from read_data_iris import read_data_all_years


def get_imerg_vod_pixel(px_lat, px_lon):
    imerg = read_data_all_years('IMERG', regridded=True, min_year=2000, max_year=2018, 
                                lon_west=px_lon-0.5, lon_east=px_lon+0.5,
                                lat_south=px_lat-0.5, lat_north=px_lat+0.5)
    imerg_lat_idx = np.argmin(np.abs(imerg.coord('latitude').points - px_lat))
    imerg_lon_idx = np.argmin(np.abs(imerg.coord('longitude').points - px_lon))
    imerg_px = imerg[:, imerg_lat_idx, imerg_lon_idx].data

    imerg_px_cube = imerg[:, imerg_lat_idx, imerg_lon_idx]
    dxr = xr.DataArray.from_iris(imerg_px_cube)
    imerg_anom = dxr.groupby("time.dayofyear") - dxr.groupby("time.dayofyear").mean("time")
    imerg_anom_px = imerg_anom.data.compute()
    imerg_px_cube.units = 'unknown'
    sqrt_imerg = imerg_px_cube**0.5
    dxr = xr.DataArray.from_iris(sqrt_imerg)
    sqrt_imerg_anom = dxr.groupby("time.dayofyear") - dxr.groupby("time.dayofyear").mean("time")
    sqrt_imerg_anom_px = sqrt_imerg_anom.data.compute()


    vod = read_data_all_years('VOD', band='X', min_year=2000, max_year=2018,
                              lon_west=px_lon-0.5, lon_east=px_lon+0.5,
                              lat_south=px_lat-0.5, lat_north=px_lat+0.5,
                              mask_surface_water=True)
    vod_lat_idx = np.argmin(np.abs(vod.coord('latitude').points - px_lat))
    vod_lon_idx = np.argmin(np.abs(vod.coord('longitude').points - px_lon))
    vod_px = vod[:, vod_lat_idx, vod_lon_idx].data

    vod_px_cube = vod[:, vod_lat_idx, vod_lon_idx]
    dxr = xr.DataArray.from_iris(vod_px_cube)
    vod_anom = dxr.groupby("time.dayofyear") - dxr.groupby("time.dayofyear").mean("time")
    vod_anom_px = vod_anom.data.compute()

    imerg_buffer = np.ones((vod_anom_px.size - imerg_px.size))*np.nan
    imerg_pad = np.concatenate((imerg_buffer, imerg_px))
    imerg_anom_pad = np.concatenate((imerg_buffer, imerg_anom_px))
    imerg_sqrt_anom_pad = np.concatenate((imerg_buffer, sqrt_imerg_anom_px))

    return imerg_pad, imerg_anom_pad, imerg_sqrt_anom_pad, vod_px, vod_anom_px


def save_pixel_time_series(px_lat, px_lon, px_desc):
    print(f'saving {px_desc}')
    start = time.time()
    imerg, imerg_anom, imerg_sqrt_anom, vod, vod_anom = get_imerg_vod_pixel(px_lat, px_lon)
    save_directory = f'../data/pixel_time_series/{px_desc}'
    os.system(f'mkdir -p {save_directory}')
    np.save(f'{save_directory}/imerg_{px_desc}.npy', ma.filled(imerg, np.nan))
    np.save(f'{save_directory}/imerg_anom_{px_desc}.npy', imerg_anom)
    np.save(f'{save_directory}/imerg_sqrt_anom_{px_desc}.npy', imerg_sqrt_anom)
    np.save(f'{save_directory}/vod_{px_desc}.npy', ma.filled(vod, np.nan))
    np.save(f'{save_directory}/vod_anom_{px_desc}.npy', vod_anom)
    end = time.time()
    print(f'saved in {(end-start)/60.:.2f} minutes')


if __name__ == '__main__':
    save_pixel_time_series(-24.625, 125.375, 'australia_3dlagDJFnonzero')
    save_pixel_time_series(3.875, 31.875, 'east_africa_20dlagMAM')
    save_pixel_time_series(-18.625, 47.375, 'madagascar_-22dlagMAM')