# precipitation-VOD-ISV
Code for Harris et al. "Satellite-Observed Vegetation Responses to Intraseasonal Precipitation Variability"

This repository contains code to reproduce the analysis and figures of Harris et al. "Satellite-Observed Vegetation Responses to Intraseasonal Precipitation Variability" (submitted to Geophysical Research Letters).

Note that this analysis relies upon Fortran code containing an algorithm for the Lomb-Scargle periodogram, from:
>Weedon et al. (2015). Evaluating the performance of hydrological models via cross-spectral analysis: case study of the Thames Basin, United Kingdom. *Journal of Hydrometeorology*, **16**(1), 214--231.

This code is not included here, but it is freely available for to anyone to use from the Met Office Science Repository Service at [https://code.metoffice.gov.uk/trac/lmed/browser#main/trunk/benchmarking](https://code.metoffice.gov.uk/trac/lmed/browser#main/trunk/benchmarking) (last access: 29 June 2022). Access requires registration for an account and this will be supported by a member of the JULES group. Requests for new accounts can be made by emailing [Jules-Support@metoffice.gov.uk](mailto:Jules-Support@metoffice.gov.uk) with details of the user’s name, email address, institution and purpose for requiring access.

# Cross-spectral analysis

The code can be used to perform the cross-spectral analysis described in Section 2 of the paper as follows:
- Obtain the Fortran code containing the Lomb-Scargle periodogram algorithm from Weedon et al. (2015) (csagan1.1.f from the Met Office repository described above).
- Enable multiprocessing with [csagan_multiprocess.py](src/csagan_multiprocess.py) following steps in [multiprocessing_setup.txt](multiprocessing_setup.txt)
- Compile csagan (requires linking to netCDF libraries)
- Run bash script [csa_multiprocess_tiles](src/csa_multiprocess_tiles). This will run the cross-spectral analysis for all pixels/seasons, saving the results in three latitude-band tiles for each season. The save directory is set in [csagan_multiprocess.py](src/csagan_multiprocess.py). A single season takes approximately 8 hours to process and produces ~3--5 GB of saved data.
- Run [significant_coherent_intraseasonal_relationships.py](src/significant_coherent_intraseasonal_relationships.py) for each tile ('tropics', 'northern', 'southern'), season and frequency-of-variability band. These variables have to be set at the top of the .py file before running. This computes and saves details of the coherent relationships that are 95% significant based on the 3-neighbour condition described in Section 2 of the paper.

# Figures
- **Figure 1**: produced by running [csa_illustration.py](src/csa_illustration.py) (requires Lomb-Scargle Fortran code)
- **Figure 2**: [lag_subplots.py](src/lag_subplots.py)
- **Figure 3** and **Figure S6**: [lag_by_land_cover.py](src/lag_by_land_cover.py)
- **Figure 4**: first run [isv_events_surface_water_tiles.py](src/isv_events_surface_water_tiles.py) to save standardised anomalies of the observed variables and save the dates of intraseasonal precipitation events at each pixel. Then [isv_composites_surface_water_filter_tiles.py](src/isv_composites_surface_water_filter_tiles.py) generates the figure.
- **Figure S1**: Created using function filter_pixel() in [vod_surface_water_filter_best.py](src/vod_surface_water_filter_best.py) with kwarg demo_figure=True
- **Figure S2**: [inundation_mask_maps.py](src/inundation_mask_maps.py). This also saves data on the number of valid observations before and after inundation masking, which is required by [lag_by_land_cover.py](src/lag_by_land_cover.py) to plot Figure S6.
- **Figure S4**: [save_example_pixel_time_series.py](src/save_example_pixel_time_series.py) saves the time series at the example locations to file. Then plot with [example_pixel_time_series_plots_multiyear.py](src/example_pixel_time_series_plots_multiyear.py)
- **Figure S5**: [percent_rainfall_variability_in_isv.py](src/percent_rainfall_variability_in_isv.py)


# Other files
- [vod_filter_monthly_global.py](src/vod_filter_monthly_global.py) is used to mask random single-timestep spikes in the VOD data, as described in Section 2.1
- [vod_surface_water_filter_best.py](src/vod_surface_water_filter_best.py) masks inundation events from the VOD data, as described in Section 2.1
- [read_data_iris.py](src/read_data_iris.py) contains functions to conveniently read in IMERG/VODCA/SSM/SWAMPS data from file, assuming the dataset is stored in yearly files
- [read_csagan_saved_output.py](src/read_csagan_saved_output.py) contains helper functions to read the output produced by [csa_multiprocess.py](src/csa_multiprocess.py)