from read_data_iris import *
import iris.coord_categorisation
from tqdm import tqdm
from bandpass_filters import bandpass_filter_missing_data
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.feature as feat
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from plot_utils import binned_cmap


lon_west = -180
lon_east = 180
lat_south= -55
lat_north = 55
imerg = read_data_all_years('IMERG', regridded=True, 
                            lon_west=lon_west, lon_east=lon_east,
                            lat_south=lat_south, lat_north=lat_north,
                            min_year=2000, max_year=2018)
imerg.units = 'unknown' #so can take sqrt without error
sqrt_imerg = imerg**0.5

iris.coord_categorisation.add_day_of_year(sqrt_imerg, 'time')
daily_means = sqrt_imerg.aggregated_by('day_of_year', iris.analysis.MEAN)
imerg_time = sqrt_imerg.coord('time')
anomaly_data = sqrt_imerg.data
daily_mean_data = daily_means.data
for t in range(imerg_time.points.size):
    doy = sqrt_imerg.coord('day_of_year').points[t]
    anomaly_data[t, :, :] -= daily_mean_data[doy-1, :, :]

# download "ESA-CCI-SOILMOISTURE-LAND_AND_RAINFOREST_MASK-fv04.2.nc" from https://doi.org/10.5281/zenodo.2575598
cci_land_mask_filename = "/prj/nceo/bethar/VODCA_global/ESA-CCI-SOILMOISTURE-LAND_AND_RAINFOREST_MASK-fv04.2.nc"
land_mask = iris.load_cube(cci_land_mask_filename,'land')
land_region = crop_cube(land_mask, lon_west, lon_east, lat_south, lat_north)
land = iris.util.reverse(land_region, land_region.coords('latitude')).data

anomaly_data = ma.filled(anomaly_data, np.nan)
lats = imerg.coord('latitude').points
lons = imerg.coord('longitude').points
total_lons = lons.size
total_lats = lats.size
band_variance = np.empty((total_lats, total_lons))
band_variance_percent = np.empty((total_lats, total_lons))
for i in tqdm(range(total_lats)):
    for j in range(total_lons):
        if land[i, j]:
            time_series = anomaly_data[:, i, j]
            filtered_series = bandpass_filter_missing_data(time_series, 1./60., 1./25., 1., order=5, min_slice_size=365)
            band_variance[i, j] = np.nanvar(filtered_series)
            band_variance_percent[i,j] = 100.*np.nanvar(filtered_series)/np.nanvar(time_series)
        else:
            band_variance[i,j] = np.nan
            band_variance_percent[i,j] = np.nan

cbar_ticks = np.arange(1, 14, 2)
cmap, norm = binned_cmap(cbar_ticks, 'plasma', extend='max')
projection = ccrs.PlateCarree()
fig = plt.figure(figsize=(15, 5)) 
ax = plt.axes(projection=projection)
p = plt.contourf(lons, lats, band_variance_percent, cbar_ticks, extend='max', cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
for c in p.collections:
    c.set_edgecolor("face")
cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.12, ax.get_position().width, 0.03])
cbar = fig.colorbar(p, orientation='horizontal', cax=cax, aspect=40, pad=0.12)
cbar.set_ticks(cbar_ticks)
cbar.ax.set_xlabel('% precipitation variance 25-60 days', fontsize=18)
cbar.ax.tick_params(labelsize=16)
ax.coastlines(color='black', linewidth=1)
ax.set_xticks(np.arange(-180, 181, 90), crs=projection)
ax.set_yticks(np.arange(-50, 51, 50), crs=projection)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params(labelsize=16)
ax.tick_params(axis='x', pad=5)
plt.savefig(f'../figures/precip_variability/percent_variance_25-60_days_land_55NS.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.savefig(f'../figures/precip_variability/percent_variance_25-60_days_land_55NS.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.show()
