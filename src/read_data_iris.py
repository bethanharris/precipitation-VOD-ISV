import iris
from iris.util import equalise_attributes
import os
import numpy as np
import xarray as xr
import cf_units
import dask.array as da
from tqdm import tqdm
from scipy.stats import linregress


def data_directory(dataset, band=None, regridded=False, mask_surface_water=False):
    if dataset == 'VOD' and band is None:
        raise ValueError('Must supply band for VOD data')
    surface_water_dir = mask_surface_water * 'filtered_surface_water/'
    directories = {'IMERG': '/localscratch/wllf029/bethar/IMERG',
                   'VOD': f'/prj/nceo/bethar/VODCA_global/filtered/{surface_water_dir}{band}-band',
                   'SM': '/prj/swift/ESA_CCI_SM/year_files_v6.1_combined_GLOBAL/',
                   'IMERG_regridded': '/prj/nceo/bethar/IMERG/regrid_p25_global',
                   'SWAMPS': '/prj/nceo/bethar/SWAMPS_daily',
                   'NDVI': '/prj/nceo/bethar/MODIS-NDVI-16day'}
    dataset_key = dataset+'_regridded' if regridded else dataset
    directory = directories[dataset_key]
    return directory
     
    
def data_filename(dataset, year, band=None, modis_sensor=None, regridded=False, mask_surface_water=False):
    if dataset == 'VOD' and band is None:
        raise ValueError('Must supply band for VOD data')
    if dataset == 'NDVI' and modis_sensor is None:
        raise ValueError('Must supply MODIS sensor (aqua/terra) for NDVI data')
    surface_water_fname = mask_surface_water * 'surface_water_'
    filenames = {'IMERG': f'IMERG.V06.{year}.daily.nc4',
                 'VOD': f'VOD-{band}-band_filtered_{surface_water_fname}{year}.nc',
                 'SM': f'{year}_volumetric_soil_moisture_daily.nc',
                 'IMERG_regridded': f'IMERG.V06.{year}.daily_p25.nc',
                 'SWAMPS': f'SWAMPS-{year}.nc',
                 'NDVI': f'/prj/nceo/bethar/MODIS-NDVI-16day/modis_{modis_sensor}_16-day_ndvi_0p25_{year}.nc'}
    dataset_key = dataset+'_regridded' if regridded else dataset
    file_path = os.path.join(data_directory(dataset, band=band, regridded=regridded, mask_surface_water=mask_surface_water), filenames[dataset_key])
    return file_path


def crop_cube(cube, lon_west, lon_east, lat_south, lat_north):
    return cube.extract(iris.Constraint(latitude=lambda cell: lat_south-1e-6 < cell < lat_north+1e-6,
                                        longitude=lambda cell: lon_west-1e-6 < cell < lon_east+1e-6))
                         

def pixel_top_left_to_centre(cube):
    lats = cube.coord('latitude').points
    resolution = lats[1] - lats[0]
    new_lats = cube.coord('latitude').copy(lats - 0.5*resolution)
    cube.replace_coord(new_lats)
    lons = cube.coord('longitude').points
    new_lons = cube.coord('longitude').copy(lons + 0.5*resolution)
    cube.replace_coord(new_lons)


def roll_cube_longitude(cube):
    """Takes a cube which goes longitude 0-360 back to -180-180."""
    lon = cube.coord('longitude')
    cube.data = np.roll(cube.data, len(lon.points) // 2)
    new_lons = lon.copy(lon.points - 180.)
    cube.replace_coord(new_lons)
    return cube


def fix_coords(cube):
    calendar = cube.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    cube.coord('time').convert_units(common_time_unit)
    times = cube.coord('time').points
    if np.all(np.abs(times.astype(int) - times) < 1e-6): #put all times at 12Z so bounds fill day
        new_time = cube.coord('time').copy(times + 0.5)
        cube.replace_coord(new_time)

    if np.all(np.diff(cube.coord('latitude').points)<0.): #make all latitude coordinate arrays ascending
        cube = iris.util.reverse(cube, cube.coords('latitude'))

    if cube.standard_name in ['air_temperature', 'land_binary_mask']: #ERA5 coordinates at top-left rather than centre of grid box
        pixel_top_left_to_centre(cube)

    if cube.coord('longitude').points.max() > 181.: #longitude on 0 to 360 instead of -180 to 180
        cube = roll_cube_longitude(cube)

    for coord_key in ['time', 'latitude', 'longitude']:
        if cube.coord(coord_key).points.size > 1:
            cube.coord(coord_key).bounds = None
            cube.coord(coord_key).guess_bounds()
            cube.coord(coord_key).bounds = np.round(cube.coord(coord_key).bounds, 3)
            cube.coord(coord_key).points = np.round(cube.coord(coord_key).points, 3)
    return cube
    

def read_land_sea_mask(lon_west=-180, lon_east=180, lat_south=-90, lat_north=90, regrid_cube=None):
    filename = '/prj/nceo/bethar/ERA5/T2m/era5-land-sea-mask.nc'
    land_sea_data = iris.load_cube(filename)[0]
    land_sea_data = fix_coords(land_sea_data)
    if regrid_cube:
        regrid_scheme = iris.analysis.AreaWeighted(mdtol=0.5)
        land_sea_data = land_sea_data.regrid(regrid_cube, regrid_scheme)
    land_sea_data_crop = crop_cube(land_sea_data, lon_west, lon_east, lat_south, lat_north)
    return land_sea_data_crop.data

    
def read_data_year(dataset, year, band=None, modis_sensor=None, regridded=False, mask_surface_water=False,
                   regrid_cube=None, lon_west=-180, lon_east=180, lat_south=-30, lat_north=30):
    if dataset == 'VOD' and band is None:
        raise ValueError('Must supply band for VOD data')
    if dataset == 'NDVI' and modis_sensor is None:
        raise ValueError('Must supply MODIS sensor (aqua/terra) for NDVI data')
    field_keys = {'IMERG': 'precipitationCal',
                  'VOD': 'vod',
                  'SM': 'sm',
                  'SWAMPS': 'frac_surface_water',
                  'NDVI': 'NDVI'}
    filename = data_filename(dataset, year, band=band, modis_sensor=modis_sensor, regridded=regridded, 
                             mask_surface_water=mask_surface_water)
    data_cube = iris.load_cube(filename, field_keys[dataset])
    data_fix = fix_coords(data_cube)
    data_crop = crop_cube(data_fix, lon_west, lon_east, lat_south, lat_north)
    if regrid_cube:
        regrid_scheme = iris.analysis.AreaWeighted(mdtol=0.5)
        data_crop = data_crop.regrid(regrid_cube, regrid_scheme)
    return data_crop
     

def read_data_all_years(dataset, band=None, modis_sensor=None, regridded=False,
                        mask_surface_water=False, regrid_cube=None,
                        min_year=2002, max_year=2016,
                        lon_west=-180, lon_east=180, lat_south=-30, lat_north=30):
    years = np.arange(min_year, max_year + 1, dtype=np.int16)
    all_years = iris.cube.CubeList([read_data_year(dataset, year, band=band, modis_sensor=modis_sensor,
                                                   mask_surface_water=mask_surface_water,
                                                   lon_west=lon_west, lon_east=lon_east,
                                                   regridded=regridded, regrid_cube=regrid_cube,
                                                   lat_south=lat_south, lat_north=lat_north) for year in years])
    equalise_attributes(all_years)
    all_data = all_years.concatenate_cube()
    if dataset in ['IMERG', 'NDVI']:
        all_data.transpose([0, 2, 1])
    return all_data


def write_regridded_datasets(band, lon_west=-180, lon_east=180, lat_south=-30, lat_north=30, 
                             min_year=2002, max_year=2016):
    vod = read_data_all_years('VOD', band=band, min_year=min_year, max_year=max_year,
                              lon_west=lon_west, lon_east=lon_east, lat_south=lat_south, lat_north=lat_north)
    years = np.arange(min_year, max_year + 1, dtype=np.int16)
    for year in tqdm(years):
        imerg_year = read_data_year('IMERG', year, regrid_cube=vod,
                                    lon_west=lon_west, lon_east=lon_east, lat_south=lat_south, lat_north=lat_north)
        iris.save(imerg_year, f'/prj/nceo/bethar/IMERG/regrid_p25_global/IMERG.V06.{year}.daily_p25.nc')
        
        
def all_regridded_datasets(band, lon_west=-180, lon_east=180, lat_south=-30, lat_north=30, 
                           min_year=2002, max_year=2016):
    vod = read_data_all_years('VOD', band=band, min_year=min_year, max_year=max_year,
                              lon_west=lon_west, lon_east=lon_east, lat_south=lat_south, lat_north=lat_north)
    imerg = read_data_all_years('IMERG', band=band, regridded=True, min_year=min_year, max_year=max_year,
                                 lon_west=lon_west, lon_east=lon_east, lat_south=lat_south, lat_north=lat_north)
    return vod, chirps, imerg


def detrend_missing_values(data):
    x = np.arange(data.size)
    valid_idx = ~np.logical_or(data>998, data<-998)
    if valid_idx.sum() > 0:
        m, b, r_val, p_val, std_err = linregress(x[valid_idx], data[valid_idx])
        detrended_data = data - (m*x + b)
    else:
        detrended_data = data
    return detrended_data


def detrend_cube(cube, dimension='time'):
    """
    Adapted from esmvalcore to work with missing values.
    Detrend data along a given dimension.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    dimension: str
        Dimension to detrend
    method: str
        Method to detrend. Available: linear, constant. See documentation of
        'scipy.signal.detrend' for details

    Returns
    -------
    iris.cube.Cube
        Detrended cube
    """
    coord = cube.coord(dimension)
    axis = cube.coord_dims(coord)[0]
    detrended = da.apply_along_axis(
        detrend_missing_values,
        axis=axis,
        arr=cube.lazy_data().rechunk([45, 500, 500]),
        shape=(cube.shape[axis],)
    )
    return cube.copy(detrended)


def monthly_anomalies(cube, detrend=False):
    if detrend:
        cube = detrend_cube(cube)
    dxr = xr.DataArray.from_iris(cube)
    climatology = dxr.groupby("time.month").mean("time")
    anomalies = (dxr.groupby("time.month") - climatology).to_iris()
    anomalies.standard_name = cube.standard_name
    anomalies.long_name = cube.long_name
    anomalies.units = cube.units
    calendar = anomalies.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    anomalies.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        anomalies.coord(coord_key).bounds = None
        anomalies.coord(coord_key).guess_bounds()
        anomalies.coord(coord_key).bounds = np.round(anomalies.coord(coord_key).bounds, 3)
        anomalies.coord(coord_key).points = np.round(anomalies.coord(coord_key).points, 3)
    return anomalies


def daily_anomalies(cube, detrend=False):
    if detrend:
        cube = detrend_cube(cube)
    dxr = xr.DataArray.from_iris(cube)
    month_day_str = xr.DataArray(dxr.indexes['time'].strftime('%m-%d'), coords=dxr.coords['time'].coords, name='month_day_str')
    climatology = dxr.groupby(month_day_str).mean("time")
    anomalies = (dxr.groupby(month_day_str) - climatology).to_iris()
    anomalies.standard_name = cube.standard_name
    anomalies.long_name = cube.long_name
    anomalies.units = cube.units
    calendar = anomalies.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    anomalies.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        anomalies.coord(coord_key).bounds = None
        anomalies.coord(coord_key).guess_bounds()
        anomalies.coord(coord_key).bounds = np.round(anomalies.coord(coord_key).bounds, 3)
        anomalies.coord(coord_key).points = np.round(anomalies.coord(coord_key).points, 3)
    return anomalies


def monthly_anomalies_normalised(cube, detrend=False):
    if detrend:
        cube = detrend_cube(cube)
    dxr = xr.DataArray.from_iris(cube)
    climatology_mean = dxr.groupby("time.month").mean("time")
    climatology_std = dxr.groupby("time.month").std("time")
    anomalies = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        dxr.groupby("time.month"),
        climatology_mean,
        climatology_std, dask='allowed'
    ).to_iris()
    anomalies.standard_name = cube.standard_name
    anomalies.long_name = cube.long_name
    anomalies.units = cube.units
    calendar = anomalies.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    anomalies.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        anomalies.coord(coord_key).bounds = None
        anomalies.coord(coord_key).guess_bounds()
        anomalies.coord(coord_key).bounds = np.round(anomalies.coord(coord_key).bounds, 3)
        anomalies.coord(coord_key).points = np.round(anomalies.coord(coord_key).points, 3)
    return anomalies


def daily_anomalies_normalised(cube, detrend=False):
    if detrend:
        cube = detrend_cube(cube)
    dxr = xr.DataArray.from_iris(cube)
    month_day_str = xr.DataArray(dxr.indexes['time'].strftime('%m-%d'), coords=dxr.coords['time'].coords, name='month_day_str')
    climatology_mean = dxr.groupby(month_day_str).mean("time")
    climatology_std = dxr.groupby(month_day_str).std("time")
    anomalies = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        dxr.groupby(month_day_str),
        climatology_mean,
        climatology_std, dask='allowed'
    ).to_iris()
    anomalies.standard_name = cube.standard_name
    anomalies.long_name = cube.long_name
    anomalies.units = cube.units
    calendar = anomalies.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    anomalies.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        anomalies.coord(coord_key).bounds = None
        anomalies.coord(coord_key).guess_bounds()
        anomalies.coord(coord_key).bounds = np.round(anomalies.coord(coord_key).bounds, 3)
        anomalies.coord(coord_key).points = np.round(anomalies.coord(coord_key).points, 3)
    return anomalies
