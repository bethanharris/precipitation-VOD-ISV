import os
from tqdm import tqdm
import numpy as np
import numpy.ma as ma
from read_data_iris import read_data_all_years
import iris.coord_categorisation
from iris.time import PartialDateTime
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import AxesGrid
from plot_utils import binned_cmap
from lag_by_land_cover import copernicus_land_cover


def save_number_seasonal_vod_obs(vod_no_sw_mask, vod_sw_mask, season):
    save_dir = '../data/number_vod_obs'
    os.system(f'mkdir -p {save_dir}')
    vod_no_sw_mask_season = vod_no_sw_mask.extract(iris.Constraint(clim_season=season.lower()))
    vod_sw_mask_season = vod_sw_mask.extract(iris.Constraint(clim_season=season.lower()))
    total_obs_no_sw_mask = vod_no_sw_mask_season.collapsed('time', iris.analysis.COUNT,
                                                           function=lambda values: values > -99.)
    total_obs_sw_mask = vod_sw_mask_season.collapsed('time', iris.analysis.COUNT,
                                                     function=lambda values: values > -99.)
    total_possible_obs = vod_no_sw_mask_season.coord('time').points.size
    np.save(f'{save_dir}/total_possible_obs_{season}.npy', total_possible_obs)
    np.save(f'{save_dir}/total_obs_no_sw_mask_{season}.npy', ma.filled(total_obs_no_sw_mask.data, 0.))
    np.save(f'{save_dir}/total_obs_sw_mask_{season}.npy', ma.filled(total_obs_sw_mask.data, 0.))


def save_ssm_seasonal_obs_numbers():
    ssm = read_data_all_years('SM', lon_west=-180, lon_east=180, 
                              lat_south=-55, lat_north=55, min_year=2000, max_year=2018)
    jun2000 = PartialDateTime(year=2000, month=6)
    dec2018 = PartialDateTime(year=2018, month=31)
    date_range = iris.Constraint(time=lambda cell: jun2000 <= cell.point <= dec2018)
    ssm = ssm.extract(date_range)
    iris.coord_categorisation.add_season(ssm, 'time', name='clim_season')
    seasons = ['MAM', 'JJA', 'SON', 'DJF']
    for season in tqdm(seasons, desc='saving SSM seasonal obs numbers'):
        save_dir = '../data/number_vod_obs'
        os.system(f'mkdir -p {save_dir}')
        ssm_season = ssm.extract(iris.Constraint(clim_season=season.lower()))
        total_obs_ssm = ssm_season.collapsed('time', iris.analysis.COUNT,
                                             function=lambda values: values > -99.)
        np.save(f'{save_dir}/total_obs_ssm_{season}.npy', ma.filled(total_obs_ssm.data, 0.))


def save_all_seasonal_obs_numbers():
    vod_no_sw_mask = read_data_all_years('VOD', band='X', lon_west=-180, lon_east=180, 
                                         lat_south=-55, lat_north=55, min_year=2000, max_year=2018,
                                         mask_surface_water=False)
    vod_sw_mask = read_data_all_years('VOD', band='X', lon_west=-180, lon_east=180, 
                                      lat_south=-55, lat_north=55, min_year=2000, max_year=2018,
                                      mask_surface_water=True)
    jun2000 = PartialDateTime(year=2000, month=6)
    dec2018 = PartialDateTime(year=2018, month=31)
    date_range = iris.Constraint(time=lambda cell: jun2000 <= cell.point <= dec2018)
    vod_no_sw_mask = vod_no_sw_mask.extract(date_range)
    vod_sw_mask = vod_sw_mask.extract(date_range)
    iris.coord_categorisation.add_season(vod_no_sw_mask, 'time', name='clim_season')
    iris.coord_categorisation.add_season(vod_sw_mask, 'time', name='clim_season')
    seasons = ['MAM', 'JJA', 'SON', 'DJF']
    for season in tqdm(seasons, desc='saving seasonal obs numbers'):
        save_number_seasonal_vod_obs(vod_no_sw_mask, vod_sw_mask, season)
    save_ssm_seasonal_obs_numbers()


def removed_for_inundation(season):
    save_dir = '../data/number_vod_obs'
    total_possible_obs = np.load(f'{save_dir}/total_possible_obs_{season}.npy')
    total_obs_no_sw_mask = np.load(f'{save_dir}/total_obs_no_sw_mask_{season}.npy')
    total_obs_sw_mask = np.load(f'{save_dir}/total_obs_sw_mask_{season}.npy')
    percent_before_mask = 100. * total_obs_no_sw_mask / total_possible_obs
    percent_after_mask = 100. * total_obs_sw_mask / total_possible_obs
    removed_by_mask = np.logical_and(percent_before_mask>=30., percent_after_mask<30.)
    return removed_by_mask


def inundation_mask_maps():
    cci_sm_mask = '../data/ESA-CCI-SOILMOISTURE-LAND_AND_RAINFOREST_MASK-fv04.2.nc' # this file is included in VODCA dataset
    lat55 = iris.Constraint(latitude=lambda cell: -55. <= cell.point <= 55.)
    land = np.flipud(iris.load_cube(cci_sm_mask, 'land').extract(lat55).data).astype(float)
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    lons = np.arange(-180, 180, 0.25) + 0.5*0.25
    lats = np.arange(-55, 55, 0.25) + 0.5*0.25
    lon_bounds = np.hstack((lons - 0.5*0.25, np.array([lons[-1]+0.5*0.25])))
    lat_bounds = np.hstack((lats - 0.5*0.25, np.array([lats[-1]+0.5*0.25])))
    fig = plt.figure(figsize=(16, 8))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(2, 2),
                    axes_pad=0.2,
                    cbar_location='bottom',
                    cbar_mode='single',
                    cbar_pad=0.15,
                    cbar_size='10%',
                    label_mode='')  # note the empty label_mode

    seasons = ['MAM', 'JJA', 'SON', 'DJF']

    for i, ax in enumerate(axgr):
        obs_removed = removed_for_inundation(seasons[i]).astype(float)
        ssm_obs = np.load(f'../data/number_vod_obs/total_obs_ssm_{seasons[i]}.npy')
        total_days = np.load(f'../data/number_vod_obs/total_possible_obs_{seasons[i]}.npy')
        ssm_obs_pc = 100. * ssm_obs / total_days
        ssm_obs_pc[land==0.] = np.nan
        obs_removed[obs_removed==0.] = np.nan
        obs_removed[np.logical_and(obs_removed==1, ssm_obs_pc==0)] = 2
        ax.coastlines(color='#999999',linewidth=0.1)
        ax.text(0.015, 0.825, f'{seasons[i]}', fontsize=16, transform=ax.transAxes)
        p = ax.pcolormesh(lon_bounds, lat_bounds, obs_removed, transform=ccrs.PlateCarree(),
                          cmap=mpl.colors.ListedColormap(['#007dea', '#b2b2b2']), rasterized=True)
        ax.set_extent((-180, 180, -55, 55), crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(-90, 91, 90), crs=projection)
        ax.set_yticks(np.arange(-50, 51, 50), crs=projection)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize=14)
        ax.tick_params(axis='x', pad=5)
    axes = np.reshape(axgr, axgr.get_geometry())
    for ax in axes[:-1, :].flatten():
        ax.xaxis.set_tick_params(which='both', 
                                 labelbottom=False, labeltop=False)
    for ax in axes[:, 1:].flatten():
        ax.yaxis.set_tick_params(which='both', 
                                 labelbottom=False, labeltop=False)
    axes = np.reshape(axgr, axgr.get_geometry())  
    cbar = axgr.cbar_axes[0].colorbar(p)
    cbar.set_ticks([1.25, 1.75])
    cbar.set_ticklabels(['masked', 'masked and no SSM obs'])
    cbar.ax.tick_params(labelsize=16) 
    plt.savefig('../figures/pixels_removed_by_inundation_mask.pdf', dpi=1000, bbox_inches='tight')
    plt.savefig('../figures/pixels_removed_by_inundation_mask.png', dpi=1000, bbox_inches='tight')
    plt.show()

    
if __name__ == '__main__':
    save_all_seasonal_obs_numbers()
    inundation_mask_maps()
