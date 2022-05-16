import numpy as np
import numpy.ma as ma
import multiprocessing
from multiprocessing import Pool, RawArray
import psutil
import time
import calendar
from tqdm import tqdm
from pathlib import Path
import pickle
import os
import subprocess
from subprocess import run
from netCDF4 import Dataset
from read_data_iris import read_data_all_years, read_land_sea_mask
from datetime_utils import *
import sys


init_dict = {}


variable_units = {'IMERG': 'mm/day',
                  'VOD': 'unitless'}


variable_names = {'IMERG': 'Precipitation (IMERG)',
                  'VOD': 'Vegetation optical depth'}


def pad_match_time(dates1, dates2, data1, data2):
    if (dates1.shape == dates2.shape) and np.allclose(dates1, dates2):
    	return dates1, data1, data2
    else:
    	dates = np.union1d(dates1, dates2)
    	data1_times = np.isin(dates, dates1)
    	data2_times = np.isin(dates, dates2)
    	data1_pad = np.empty((dates.size, data1.shape[1], data1.shape[2]))
    	data2_pad = np.empty((dates.size, data2.shape[1], data2.shape[2]))
    	data1_pad[:] = np.nan
    	data2_pad[:] = np.nan
    	data1_pad[data1_times, :, :] = data1
    	data2_pad[data2_times, :, :] = data2
    	return dates, data1_pad, data2_pad


def season_from_abbr(season_abbr):
    if len(season_abbr) < 2:
        raise KeyError('Use seasons longer than one month')
    rolling_months = ''.join([m[0] for m in calendar.month_abbr[1:]])*2
    if season_abbr in rolling_months:
        season_start_idx = rolling_months.find(season_abbr)
        season_end_idx = season_start_idx + len(season_abbr)
        month_list = [(m%12)+1 for m in range(season_start_idx, season_end_idx)]
    else:
      raise KeyError(f'{season_abbr} not a valid sequence of months')
    return month_list


def mask_to_months(dates, data, month_list=np.arange(12)+1):
    all_months = np.array([num2date(d, units='days since 1970-01-01', calendar='gregorian').month for d in dates])
    mask = ~np.isin(all_months, np.array(month_list))
    masked_data = np.copy(data)
    masked_data[mask] = np.nan
    return masked_data


def make_data_array(data_variable, band='Ku', mask_surface_water=False,
                    lon_west=-180, lon_east=180, lat_south=-30, lat_north=30,
                    min_year=2000, max_year=2018, month_list=np.arange(12)+1, percent_readings_required=30.):
    regridded = (data_variable == 'IMERG')
    data = read_data_all_years(data_variable, band=band, regridded=regridded, mask_surface_water=mask_surface_water,
                               min_year=min_year, max_year=max_year,
                               lon_west=lon_west, lon_east=lon_east, lat_south=lat_south, lat_north=lat_north)
    dates = data.coord('time').points
    lons = data.coord('longitude').points
    lats = data.coord('latitude').points
    data_array = ma.filled(data.data, np.nan)
    data_array_mask_season = mask_to_months(dates, data_array, month_list=month_list)
    possible_days = (~np.isnan(mask_to_months(dates, np.ones_like(data_array[:, 0, 0]), month_list=month_list))).astype(int).sum()
    number_readings = (~np.isnan(data_array_mask_season)).sum(axis=0)
    pc_readings = 100.*number_readings/possible_days
    insufficient_readings = pc_readings < percent_readings_required
    data_array_mask_season[:, insufficient_readings] = np.nan
    return dates, data_array_mask_season, lats, lons


def make_data_arrays(reference_variable, response_variable, band='Ku', mask_surface_water=False,
                     lon_west=-180, lon_east=180, lat_south=-30, lat_north=30, 
                     min_year=2002, max_year=2016, return_coords=False,
                     monthly_anomalies=False, flip_reference_sign=False, flip_response_sign=False,
                     month_list=np.arange(12)+1, percent_readings_required=30.):
    reference_dates, reference_array, lats, lons = make_data_array(reference_variable, band=band,
                                                                   mask_surface_water=mask_surface_water,
                                                                   lon_west=lon_west, lon_east=lon_east, 
                                                                   lat_south=lat_south, lat_north=lat_north,
                                                                   min_year=min_year, max_year=max_year, month_list=month_list,
                                                                   percent_readings_required=percent_readings_required)
    response_dates, response_array, lats, lons = make_data_array(response_variable, band=band, 
                                                                 mask_surface_water=mask_surface_water,
                                                                 lon_west=lon_west, lon_east=lon_east, 
                                                                 lat_south=lat_south, lat_north=lat_north,
                                                                 min_year=min_year, max_year=max_year, month_list=month_list,
                                                                 percent_readings_required=percent_readings_required)
    common_dates, padded_reference, padded_response = pad_match_time(reference_dates, response_dates,
                                                                     reference_array, response_array)
    decimal_dates = np.array([days_since_1970_to_decimal_year(date) for date in common_dates])

    land_fraction = read_land_sea_mask(lon_west=lon_west, lon_east=lon_east, lat_south=lat_south, lat_north=lat_north)
    sea = (land_fraction == 0.)
    padded_reference[:, sea] = np.nan
    padded_response[:, sea] = np.nan

    if monthly_anomalies:
        months = [decimal_year_to_datetime(d).month for d in decimal_dates]
        reference_monthly_means = []
        response_monthly_means = []
        reference_anomaly_time_series = []
        response_anomaly_time_series = []
        for m in np.arange(12)+1:
            month_idcs = np.where(np.array(months)==m)[0]
            references_month = padded_reference[month_idcs]
            reference_monthly_means.append(ma.mean(ma.masked_invalid(references_month)))
            responses_month = padded_response[month_idcs]
            response_monthly_means.append(ma.mean(ma.masked_invalid(responses_month)))
        for i in range(padded_reference.size):
            month = months[i]
            reference_mean_month = reference_monthly_means[month-1]
            response_mean_month = response_monthly_means[month-1]
            reference_anomaly_time_series.append(padded_reference[i] - reference_mean_month)
            response_anomaly_time_series.append(padded_response[i] - response_mean_month)
        padded_reference = np.array(reference_anomaly_time_series)
        padded_response = np.array(response_anomaly_time_series)

    if flip_reference_sign:
        padded_reference *= -1.
    if flip_response_sign:
        padded_response *= -1.
    
    if return_coords:
        return decimal_dates, padded_reference, padded_response, lats, lons
    else:
        return decimal_dates, padded_reference, padded_response


def create_input_file(reference_variable, response_variable, dates, reference_data, response_data,
                      save_filename):
    valid_response_idcs = ~np.isnan(response_data)
    valid_reference_idcs = ~np.isnan(reference_data)
    valid_idcs = np.logical_and(valid_response_idcs, valid_reference_idcs)
    valid_dates = dates[valid_idcs]
    valid_references = reference_data[valid_idcs].data
    valid_responses = response_data[valid_idcs].data
    
    f = Dataset(save_filename, 'w', format='NETCDF4')
    f.description = f'{variable_names[reference_variable]} and {variable_names[response_variable]} daily time series'
    today = datetime.today()
    f.history = "Created " + today.strftime("%d/%m/%y")
    f.createDimension('time', None)
    days = f.createVariable('Time', 'f8', 'time')
    days[:] = valid_dates
    days.units = 'years'
    data_response = f.createVariable(f'{response_variable}', 'f4', ('time'))
    data_response.units = variable_units[response_variable]
    data_reference = f.createVariable(f'{reference_variable}', 'f4', ('time'))
    data_reference.units = variable_units[reference_variable]
    data_response[:] = valid_responses
    data_reference[:] = valid_references
    f.close()



def run_csagan(exe_filename, data_filename, netcdf, time_variable_name, 
               time_format, time_double, obs_variable_name, model_variable_name,
               frequency_units, ray_freq, model_units, time_earliest, pre_whiten, correct_bias):
                          
               netcdf = str(netcdf + 1) #convert from False/True to 1/2
               time_variable_codes = {'continuous': '1', 'integer seconds relative': '2', 'fixed time step': '3'}
               time_format = time_variable_codes[time_format]
               frequency_unit_codes = {'year_year': '1', 'day_day': '2', 'year_day': '3', 'hour_hour': '4',
                                       'minute_minute': '5', 'second_second': 6}
               frequency_units = frequency_unit_codes[frequency_units]
               change_ray_freq = 'Y' if ray_freq else 'N'
               time_double = str(time_double + 1)
               correct_bias = 'Y' if correct_bias else 'N'
               time_earliest = 'E' if time_earliest else 'L'
               pre_whiten = str(int(pre_whiten)) # string 0 or 1

               process_id = data_filename.split('-')[-1].split('.')[0]

               csagan_args = os.linesep.join([netcdf, process_id, data_filename, time_variable_name, time_format,
                                              time_double, obs_variable_name, model_variable_name, frequency_units,
                                              model_units, time_earliest, change_ray_freq, correct_bias])

               csagan = run(exe_filename, text=True, input=csagan_args, stdout=subprocess.DEVNULL)#, stderr=subprocess.DEVNULL)
                   
                   
def default_run(exe_filename, reference_variable, response_variable, data_filename):
    netcdf = True
    time_variable_name = 'Time'
    time_format = 'continuous'
    time_double = True
    obs_variable_name = reference_variable
    model_variable_name = response_variable
    frequency_units = 'year_day'
    ray_freq = False
    model_units = 'unknown'
    time_earliest = True
    pre_whiten = False
    correct_bias = False
    run_csagan(exe_filename, data_filename, netcdf, time_variable_name, 
               time_format, time_double, obs_variable_name, model_variable_name,
               frequency_units, ray_freq, model_units, time_earliest, pre_whiten, correct_bias)
    process_id = int(data_filename.split('-')[-1].split('.')[0])    
    spectra_results = read_csagan_output(process_id)
    return spectra_results
                   
    
def delete_csagan_output(process_id, directory='.'):
    os.remove(f'{directory}/csaout-{process_id}.nc')
    os.remove(f'{directory}/csaout-phase95-{process_id}.nc')


def delete_csagan_input(process_id, reference_variable, response_variable, directory='.'):
    os.remove(f'{directory}/{reference_variable}_{response_variable}_input-{process_id}.nc')
    

def read_csagan_output(process_id, directory='.'):
    spectra_filename = f'{directory}/csaout-{process_id}.nc'
    spectra_results = {}
    with Dataset(spectra_filename, 'r') as spectra_data:
        spectra_results['resolution_bandwidth'] = spectra_data.getncattr('Resolution_bandwidth')
        spectra_results['period'] = spectra_data.variables['period'][:]
        spectra_results['log_power_obs'] = spectra_data.variables['logpobs'][:]
        spectra_results['log_power_model'] = spectra_data.variables['logpmod'][:]
        spectra_results['coherency'] = spectra_data.variables['coherency'][:]
        spectra_results['phase'] = spectra_data.variables['phase'][:]
        spectra_results['phase_upper95'] = spectra_data.variables['ph95upr'][:]
        spectra_results['phase_lower95'] = spectra_data.variables['ph95lor'][:]
        spectra_results['amplitude'] = spectra_data.variables['amratio'][:]
        spectra_results['amplitude_upper95'] = spectra_data.variables['ar95upr'][:]
        spectra_results['amplitude_lower95'] = spectra_data.variables['ar95lor'][:]
    return spectra_results
    

def reference_response_spectra(exe_filename, process_id, reference_variable, response_variable,
                               dates, reference_data, response_data):
    output_directory = os.path.dirname(exe_filename)
    if output_directory == '':
        output_directory = '.'
    data_filename = f'{output_directory}/{reference_variable}_{response_variable}_input-{process_id}.nc'
    try:
        create_input_file(reference_variable, response_variable, dates, reference_data, response_data,
                          data_filename)
        spectra_results = default_run(exe_filename, reference_variable, response_variable, data_filename)
        delete_csagan_output(process_id, directory='.')
        delete_csagan_input(process_id, reference_variable, response_variable, directory=output_directory)
    except:
        print(f'***NO SPECTRA CREATED FOR PIXEL {process_id}***')
        spectra_results = {}
    return spectra_results


def csa_from_indices(coords):
    response_variable = init_dict['response_variable']
    reference_variable = init_dict['reference_variable']
    decimal_dates = np.frombuffer(init_dict['dates']).reshape(init_dict['dates_shape'])
    reference_array = np.frombuffer(init_dict['reference']).reshape(init_dict['reference_shape'])
    response_array = np.frombuffer(init_dict['response']).reshape(init_dict['response_shape'])
    lat_idx, lon_idx = coords
    readings_reference = (~np.isnan(reference_array[:, lat_idx, lon_idx])).astype(int).sum()
    readings_response = (~np.isnan(response_array[:, lat_idx, lon_idx])).astype(int).sum()
    sufficient_readings = (readings_reference > 0.) and (readings_response > 0.)
    if sufficient_readings:
        px_ids = np.frombuffer(init_dict['px_id']).reshape(init_dict['px_id_shape'])
        process_id = int(px_ids[lat_idx, lon_idx])
        d = '/users/global/bethar/python/precipitation-VOD-ISV'
        spectra = reference_response_spectra(f'{d}/csagan/csagan-multiprocess-updated.x', process_id,
                                             reference_variable, response_variable,
                	                         decimal_dates, reference_array[:, lat_idx, lon_idx],
                                             response_array[:, lat_idx, lon_idx])
    else:
        spectra = {}
    return spectra


def make_shared_array(data_array, dtype=np.float64):
    data_shared = RawArray('d', data_array.size)
    data_shared_np = np.frombuffer(data_shared, dtype=dtype).reshape(data_array.shape)
    np.copyto(data_shared_np, data_array)
    return data_shared, data_array.shape


def init_worker(reference_variable, response_variable, decimal_dates, dates_shape,
                reference_array, reference_shape, response_array, response_shape,
                px_id_array, px_id_shape):
    init_dict['reference_variable'] = reference_variable
    init_dict['response_variable'] = response_variable
    init_dict['dates'] = decimal_dates
    init_dict['dates_shape'] = dates_shape
    init_dict['reference'] = reference_array
    init_dict['reference_shape'] = reference_shape
    init_dict['response'] = response_array
    init_dict['response_shape'] = response_shape
    init_dict['px_id'] = px_id_array
    init_dict['px_id_shape'] = px_id_shape


def write_to_dataset(filename, results, results_lats, results_lons):
    if 'tropics' in filename:
        lat_south = -35
        lat_north = 35
    elif 'northern' in filename:
        lat_south = 25
        lat_north = 65
    elif 'southern' in filename:
        lat_south = -60
        lat_north = -25
    elif 'polar' in filename:
        lat_south = 55
        lat_north = 80
    else:
        raise KeyError('Save filename not recognised as belonging to a defined latitude band')

    region_lats = np.arange(lat_south, lat_north, 0.25) + 0.5*0.25
    region_lons = np.arange(-180, 180, 0.25) + 0.5*0.25

    start_lat = np.argmin(np.abs(region_lats-results_lats.min()))
    end_lat = np.argmin(np.abs(region_lats-results_lats.max()))
    start_lon = np.argmin(np.abs(region_lons-results_lons.min()))
    end_lon = np.argmin(np.abs(region_lons-results_lons.max()))
 
    if Path(filename).is_file():
        region_array = pickle.load(open(filename, 'rb'))
    else:
        region_array = np.empty((region_lats.size, region_lons.size), dtype=object)
        for i in range(region_lats.size):
            for j in range(region_lons.size):
                region_array[i, j] = {}

    for i, lat_idx in enumerate(range(start_lat, end_lat+1)):
        for j, lon_idx in enumerate(range(start_lon, end_lon+1)):
            region_array[lat_idx, lon_idx] = results[i, j]

    pickle.dump(region_array, open(filename, 'wb'))


if __name__ == '__main__':
    # run e.g. ">> python csagan_multiprocess.py tropics NDJFM -180 180 -35 35" for pan-tropical analysis in NDJFM
    region_name = sys.argv[1]
    season = sys.argv[2]
    region_coords = sys.argv[3:]
    lon_west, lon_east, lat_south, lat_north = [float(coord) for coord in region_coords]
    print(f'region: {region_name}, west: {lon_west} deg, east: {lon_east} deg, south: {lat_south} deg, north: {lat_north} deg')

    reference_variable = 'IMERG'
    response_variable = 'VOD'
    band = 'X' #only relevant for VOD
    months = season_from_abbr(season)
    decimal_dates, reference_array, response_array, lats, lons = make_data_arrays(reference_variable, response_variable,
                                                                                  band=band, mask_surface_water=True,
                                                                                  lon_west=lon_west, lon_east=lon_east,
                                                                                  lat_south=lat_south, lat_north=lat_north,
                                                                                  min_year=2000, max_year=2018,
                                                                                  return_coords=True, flip_response_sign=False,
                                                                                  month_list=months,
                                                                                  percent_readings_required=30.)
   
    total_pixels = reference_array[0, :, :].size

    lat_idcs = np.arange(response_array.shape[1])
    lon_idcs = np.arange(response_array.shape[2])
    LAT, LON = np.meshgrid(lat_idcs, lon_idcs)
    coords = zip(LAT.ravel(), LON.ravel())
    px_ids = np.arange(lats.size * lons.size).astype(int).reshape(LAT.T.shape)

    dates_shared, dates_shape = make_shared_array(decimal_dates)
    reference_shared, reference_shape = make_shared_array(reference_array)
    response_shared, response_shape = make_shared_array(response_array)
    px_id_shared, px_id_shape = make_shared_array(px_ids)

    shared_data = (reference_variable, response_variable, dates_shared, dates_shape,
                   reference_shared, reference_shape, response_shared, response_shape,
                   px_id_shared, px_id_shape)

    print(f'start pool: {total_pixels} pixels')
    start = time.time()
    with Pool(processes=8, initializer=init_worker, initargs=shared_data) as pool:
        csa_output = pool.map(csa_from_indices, coords, chunksize=1)
    results = np.reshape(csa_output, response_shape[1:], order='F')
    end = time.time()
    dud_pixels = (results=={}).sum(axis=1).sum(axis=0)
    print(f'completed {total_pixels} pixels in {end-start} seconds, {dud_pixels} pixels did not have enough data for computation')
    band_label = f'_{band}' if (reference_variable == 'VOD' or response_variable == 'VOD') else ''
    write_to_dataset(f'/prj/nceo/bethar/cross_spectral_analysis_results/test/{region_name}_{reference_variable}_{response_variable}_spectra{band_label}_{season}_masksw_corrected.p', results, lats, lons)
