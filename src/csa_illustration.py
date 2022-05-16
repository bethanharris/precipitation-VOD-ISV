import string
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
from csagan_multiprocess import *
from datetime_utils import datetime_to_decimal_year


DAYS = np.arange(5000.)
SHOW_DAYS = 500
PRECIP_AMPLITUDE_ANNUAL = 5.
VOD_AMPLITUDE_ANNUAL = 0.2
PRECIP_AMPLITUDE_ISV = 2.
VOD_AMPLITUDE_ISV = 0.1
ANNUAL_CYCLE_LAG = 90.
ISV_LAG = 10.

precip_colour = 'k'
vod_colour = '#F93434'


def lagged_signal(scale, offset, period, lag):
    time_series = scale * np.sin(2. * np.pi * (DAYS - lag)/period) + offset
    return time_series


def generate_noise(scale_factor, rng_seed=None):
    rng = np.random.default_rng(rng_seed)
    noise = rng.normal(loc=0., scale=scale_factor, size=DAYS.size)
    return noise


def mask_random_timesteps(time_series, mask_percentage=20, mask_chunk_length=10, rng_seed=None):
    rng = np.random.default_rng(rng_seed)
    time_series_length = time_series.size
    number_timesteps_to_mask = int(time_series_length * float(mask_percentage)/100.)
    number_chunks_to_mask = number_timesteps_to_mask//mask_chunk_length
    select_from_indices =  np.unique(np.arange(time_series_length-mask_chunk_length).astype(int)//mask_chunk_length)
    start_idx_to_remove = rng.choice(select_from_indices, replace=False, # don't choose indices where end of masked section will go out of range
                                           size=number_chunks_to_mask) * mask_chunk_length # replace kwarg stops indices being picked twice
    for s in start_idx_to_remove:
        time_series[s:s+mask_chunk_length] = np.nan
    return time_series


def all_time_series():
    precip_time_series = {}
    vod_time_series = {}
    precip_time_series['annual'] = lagged_signal(PRECIP_AMPLITUDE_ANNUAL, 0., 365., 0.)
    vod_time_series['annual'] = lagged_signal(VOD_AMPLITUDE_ANNUAL, 0., 365., ANNUAL_CYCLE_LAG)
    precip_time_series['isv'] = lagged_signal(PRECIP_AMPLITUDE_ISV, 0., 30., 0.)
    vod_time_series['isv'] = lagged_signal(VOD_AMPLITUDE_ISV, 0., 30., ISV_LAG)
    precip_time_series['noise'] = generate_noise(0.25*PRECIP_AMPLITUDE_ISV, rng_seed=1896)
    vod_time_series['noise'] = generate_noise(0.25*VOD_AMPLITUDE_ISV, rng_seed=4203)
    precip_time_series['total'] = precip_time_series['annual'] + precip_time_series['isv'] + precip_time_series['noise']
    vod_time_series['total'] = vod_time_series['annual'] + vod_time_series['isv'] + vod_time_series['noise']
    # set seed to mask the same time steps in precip and VOD time series
    precip_time_series['masked_total'] = mask_random_timesteps(precip_time_series['total'], rng_seed=2024)
    vod_time_series['masked_total'] = mask_random_timesteps(vod_time_series['total'], rng_seed=2024)
    return precip_time_series, vod_time_series


def get_spectra(precip, vod):
    base_date = datetime(2000, 1, 1)
    dates = [base_date + timedelta(days=n) for n in range(vod.size)]
    decimal_dates = np.array([datetime_to_decimal_year(d) for d in dates])
    csagan = '/users/global/bethar/python/cross-spectral-veg-precip/csagan/csagan-multiprocess.x'
    spectra = reference_response_spectra(csagan, 0, 'IMERG', 'VOD', decimal_dates, precip, vod)
    return spectra


def precip_vod_plot(precip, vod, anomaly=False, ax=None, y_labels=True):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax2 = ax.twinx()
    ax.plot(DAYS[0:SHOW_DAYS], precip[0:SHOW_DAYS], color=precip_colour)
    ax2.plot(DAYS[0:SHOW_DAYS], vod[0:SHOW_DAYS], color=vod_colour)#, linestyle=(0, (4, 2))
    precip_limit = PRECIP_AMPLITUDE_ANNUAL + PRECIP_AMPLITUDE_ISV
    vod_limit = VOD_AMPLITUDE_ANNUAL + VOD_AMPLITUDE_ISV
    ax.set_ylim([-precip_limit * 1.2, precip_limit * 1.2])
    ax2.set_ylim([-vod_limit * 1.2, vod_limit * 1.2])
    ax.set_xlabel('day', fontsize=14)
    ax.tick_params(axis='y', colors=precip_colour)
    ax.tick_params(labelsize=12)
    ax2.tick_params(axis='y', colors=vod_colour)
    ax2.tick_params(labelsize=12)
    ax.set_xlim([0, SHOW_DAYS])
    if y_labels:
        if anomaly:
            precip_label = 'precipitation\nanomaly (mm day$^{-1}$)'
            vod_label = 'VOD anomaly\n(unitless)'
        else:
            precip_label = 'precipitation\n(mm day$^{-1}$)'
            vod_label = 'VOD (unitless)'
        ax.set_ylabel(precip_label, fontsize=14, color=precip_colour)
        ax2.set_ylabel(vod_label, fontsize=14, color=vod_colour)


def plot_coherency(spectra, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.plot(spectra['period'], spectra['coherency'], 'k', linewidth=0.75)
    ax.set_xscale('log')
    ax.set_xlim([2, 5000])
    ax.set_xticks([3, 10, 30, 100, 300, 1000])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.5g'))
    ax.set_xlabel('period (days)', fontsize=14)
    ax.set_ylabel('coherency', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim([0., 1.])
    ax.axhline(0.8733, color='k', linewidth=0.75, linestyle='--', dashes=[5, 10], label='99% CL')
    ax.axhline(0.7795, color='k', linewidth=0.75, linestyle='-', dashes=[12, 6, 2, 6], label='95% CL')
    ax.legend(fontsize=13, loc='lower right')


def plot_phase_difference(spectra, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    coh95 = spectra['coherency'] > 0.7795
    phase_deg = spectra['phase'][coh95]
    phase_upper95 = spectra['phase_upper95'][coh95]
    phase_lower95 = spectra['phase_lower95'][coh95]
    period = spectra['period'][coh95]
    lag_days = phase_deg/360. * period
    lag_upper95_days = phase_upper95/360. * period
    lag_lower95_days = phase_lower95/360. * period
    lag_errors = np.stack([lag_days - lag_lower95_days, lag_upper95_days - lag_days])
    closest_to_annual = np.argmin(np.abs(period-365.))
    closest_to_isv = np.argmin(np.abs(period-30.))
    lag_annual = lag_days[closest_to_annual]
    lag_isv = lag_days[closest_to_isv]
    lag_error_annual = lag_annual - lag_lower95_days[closest_to_annual]
    lag_error_isv = lag_isv - lag_lower95_days[closest_to_isv]
    plot_in_red = [closest_to_annual, closest_to_isv]
    plot_in_black = [x for x in range(period.size) if x not in plot_in_red]
    ax.errorbar(period[plot_in_black], lag_days[plot_in_black], yerr=lag_errors[:, plot_in_black], linewidth=0, elinewidth=1,
                     marker='o', ms=3, fmt='k', ecolor='gray', capsize=0, mew=0, label='95% CI')
    ax.errorbar(period[plot_in_red], lag_days[plot_in_red], yerr=lag_errors[:, plot_in_red],
                linewidth=0, elinewidth=1, marker='o', ms=5, fmt=vod_colour, ecolor=vod_colour, capsize=0, mew=0)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.5g'))
    ax.set_xlabel('period (days)', fontsize=14)
    ax.legend(fontsize=14, loc='upper left')
    ax.set_xlim([20, 500])
    ax.set_xticks([30, 100, 300])
    ax.set_xlim([2, 5000])
    ax.set_xticks([3, 10, 30, 100, 300, 1000])
    period_in_range = np.logical_and(period >= 20, period <= 500)
    max_data = max(lag_upper95_days[period_in_range])
    min_data = min(lag_lower95_days[period_in_range])
    ax.set_ylim([1.05*min_data, 1.05*max_data])
    ax.set_ylabel('phase difference (days)', fontsize=13, labelpad=0)
    ax.annotate(f'{lag_isv:0.1f}$\\pm${lag_error_isv:0.1f}', (period[closest_to_isv], lag_isv), 
                xytext=(period[closest_to_isv], lag_isv+12),
                fontsize=14, color=vod_colour, va='center', ha='center')
    ax.annotate(f'{lag_annual:0.1f}$\\pm${lag_error_annual:0.1f}', (period[closest_to_annual], lag_annual), 
                xytext=(period[closest_to_annual]+100, lag_annual),
                fontsize=14, color=vod_colour, va='center', ha='left')
    ax.tick_params(labelsize=12)


def subplots(precip, vod, spectra):
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(4, 2, hspace=0.25, wspace=0.41)
    annual_cycle_ax = plt.subplot(gs[0, 0])
    isv_ax = plt.subplot(gs[1, 0])
    noise_ax = plt.subplot(gs[2, 0])
    time_series_ax = plt.subplot(gs[3, 0])
    coherency_ax = plt.subplot(gs[0:2, 1])
    phase_diff_ax = plt.subplot(gs[2:, 1])
    precip_vod_plot(precip['annual'], vod['annual'], ax=annual_cycle_ax, y_labels=False)
    precip_vod_plot(precip['isv'], vod['isv'], ax=isv_ax, y_labels=False)
    precip_vod_plot(precip['noise'], vod['noise'], ax=noise_ax, y_labels=False)
    precip_vod_plot(precip['masked_total'], vod['masked_total'], ax=time_series_ax, y_labels=False)
    plot_coherency(spectra, ax=coherency_ax)
    plot_phase_difference(spectra, ax=phase_diff_ax)
    for ax in [annual_cycle_ax, isv_ax, noise_ax, coherency_ax]:
        ax.set_xticklabels([])
        ax.set_xlabel('')
    all_axes = [annual_cycle_ax, isv_ax, noise_ax, time_series_ax, coherency_ax, phase_diff_ax]
    alphabet = string.ascii_lowercase
    for i, ax in enumerate(all_axes):
        ax.text(0, 1.0, f'$\\bf{{({alphabet[i]})}}$', transform=ax.transAxes, va='bottom', fontsize=12)
    fig.text(0.075, 0.5, 'precipitation anomaly (mm day$^{-1}$)', va='center', rotation='vertical', fontsize=14, color=precip_colour)
    fig.text(0.5, 0.5, 'VOD anomaly (unitless)', va='center', rotation=270, fontsize=14, color=vod_colour)
    annual_cycle_ax.set_title('annual cycle', fontsize=12, pad=0)
    isv_ax.set_title('intraseasonal variability', fontsize=12, pad=0)
    noise_ax.set_title('white noise', fontsize=12, pad=0)
    time_series_ax.set_title('total plus data gaps', fontsize=12, pad=0)
    plt.savefig('../figures/csagan_illustration/fig1/csa_illustration_missing_data_newcolours.png', bbox_inches='tight', dpi=400)
    plt.savefig('../figures/csagan_illustration/fig1/csa_illustration_missing_data_newcolours.pdf', bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == '__main__':
    precip, vod = all_time_series()
    spectra = get_spectra(precip['masked_total'], vod['masked_total'])
    subplots(precip, vod, spectra)