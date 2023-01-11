# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 21:28:59 2022

@author: David J. Kedziora
"""

import numpy as np
from lmfit import Parameters, minimize

def compile_domain_knowledge(in_pulse_freq, in_delta_zero, in_delay_range):
    
    # From pulse details, calculate expected number of peaks in the histogram.
    # Work out which peak is the zero-delay multi-photon-emission (MPE) peak.
    # Work out at which raw delay the peaks start.
    pulse_period = 1/(in_pulse_freq)
    knowledge = dict()
    knowledge["pulse_period"] = pulse_period
    knowledge["n_peaks"] = 1 + int(in_delay_range/pulse_period)
    knowledge["id_mpe"] = int(in_delta_zero/pulse_period)
    knowledge["delay_start"] = np.mod(in_delta_zero, pulse_period)
    
    return knowledge

def calc_g2zero_quick(in_df_sample, in_bin_size, in_knowledge):
    
    # Extract domain knowledge on how the event histogram should look.
    n_peaks = in_knowledge["n_peaks"]
    id_mpe = in_knowledge["id_mpe"]
    delay_start = in_knowledge["delay_start"]
    pulse_period = in_knowledge["pulse_period"]
    
    # Search for max amplitudes within 1/4 pulse period of expected locations.
    # Sample the inter-peak mid-positions for the background detection events.
    amps_raw = np.ma.array(np.zeros(n_peaks), mask=False)
    bg_samples = np.zeros(n_peaks-1)
    for i in range(n_peaks):
        search_min = np.max([round((delay_start - pulse_period/4 + i*pulse_period)
                                  /in_bin_size), 0])
        search_max = np.min([round((delay_start + pulse_period/4 + i*pulse_period)
                                  /in_bin_size), in_df_sample.size-1])
        amps_raw[i] = np.max(in_df_sample[search_min:search_max])
        if i != 0:
            id_bg_sample = round((delay_start + (i-1/2)*pulse_period)/in_bin_size)
            bg_samples[i-1] = in_df_sample[id_bg_sample]

    # For rigour, generate a 2D matrix of g2(0) estimates.
    # They will depend on the raw amplitude and raw background samples.            
    # g2zero_estimates = np.zeros((n_peaks-1)*(n_peaks-1))   
    amp_raw_mpe = amps_raw[id_mpe]
    amps_raw.mask[id_mpe] = True
    bg_sample_matrix = np.matlib.repmat(bg_samples, (n_peaks-1), 1)
    amps_raw_matrix = np.transpose(np.matlib.repmat(amps_raw.compressed(), (n_peaks-1), 1))
    g2zero_estimates = ((amp_raw_mpe - bg_sample_matrix)/(amps_raw_matrix - bg_sample_matrix)).flatten()
    g2zero_stats = {"avg": np.mean(g2zero_estimates),
                    "std": np.std(g2zero_estimates),
                    "min": np.min(g2zero_estimates),
                    "max": np.max(g2zero_estimates)}

    # Calculate background statistics, especially the average.
    bg_avg = np.mean(bg_samples)
    bg_std = np.std(bg_samples)
    bg_min = np.min(bg_samples)
    bg_max = np.max(bg_samples)
    bg_stats = {"avg": bg_avg,
                "std": bg_std,
                "min": bg_min,
                "max": bg_max}
    
    # Calculate amplitude statistics, correcting for the background.
    # Mask out the MPE peak when processing the other peaks.
    amp_mpe = amp_raw_mpe - bg_avg
    amp_avg = np.mean(amps_raw) - bg_avg
    amp_std = np.std(amps_raw)
    amp_min = np.min(amps_raw) - bg_avg
    amp_max = np.max(amps_raw) - bg_avg
    amp_stats = {"avg": amp_avg,
                 "std": amp_std,
                 "min": amp_min,
                 "max": amp_max,
                 "mpe": amp_mpe}
    
    return g2zero_stats, amp_stats, bg_stats

# Define the core fitting function for the delay histogram.
# Is based on Eq. (2) of: https://doi.org/10.1063/1.5143786
# Does not include an overall decay for the envelope.
# x: domain (delays)
# y: events per bin to be fitted
# in_knowledge: domain knowledge for how the histogram should look
# bg: background number of events per bin (noise)
# delay_shift: a minor adjustment for peak delays to get the best fit
# amp_env: envelope amplitude for the n peaks
# decay_peak: the decay scale for an individual peak
# amp_ratio: the ratio of the MPE peak to the envelope amplitude, i.e. g2(0)
def func(params, x, y, in_knowledge, return_fit_error=False):
    bg = params["bg"]
    delay_shift = params["delay_shift"]
    amp_env = params["amp_env"]
    decay_peak = params["decay_peak"]
    amp_ratio = params["amp_ratio"]
    
    # Extract domain knowledge on how the event histogram should look.
    n_peaks = in_knowledge["n_peaks"]
    id_mpe = in_knowledge["id_mpe"]
    pulse_period = in_knowledge["pulse_period"]
    delay_start = in_knowledge["delay_start"]
    
    delay_start += delay_shift  # Adjust the domain knowledge with the fit.
    
    fit = bg
    for i in range(n_peaks):
        term = amp_env*np.exp(-abs(x-delay_start-i*pulse_period)/decay_peak)
        if i == id_mpe:
            term = term*amp_ratio
        fit = fit + term
    if return_fit_error:
        return fit - y
    else:
        return fit

def calc_g2zero_fit(in_df_sample, in_df_delays, in_knowledge):
    
    pulse_period = in_knowledge["pulse_period"]
    max_events = sum(in_df_sample)
    
    params = Parameters()
    params.add("bg", value = 0, min = 0, max = max_events)
    params.add("delay_shift", value = 0, min = -pulse_period/2, max = pulse_period/2)
    params.add("amp_env", value = 1, min = 0, max = max_events)
    params.add("decay_peak", value = in_df_delays[1] - in_df_delays[0], min = 0, max = pulse_period)
    params.add("amp_ratio", value = 0.5, min = 0, max = 1)
    
    # Run a five-parameter fit with the Nelder Mead method.
    # Refine amplitude fit with the least-squares method.
    # Then run a five-parameter least-squares fit, just for the errors.
    fitted_params = minimize(func, params, args=(in_df_delays, in_df_sample, in_knowledge, True), 
                             method="nelder_mead", calc_covar=False)

    fitted_params.params["delay_shift"].vary = False
    fitted_params.params["bg"].vary = False
    fitted_params.params["decay_peak"].vary = False
    fitted_params = minimize(func, fitted_params.params, args=(in_df_delays, in_df_sample, in_knowledge, True), 
                             method="least_squares", calc_covar=False)
    
    fitted_params.params["delay_shift"].vary = True
    fitted_params.params["bg"].vary = True
    fitted_params.params["decay_peak"].vary = True
    fitted_params = minimize(func, fitted_params.params, args=(in_df_delays, in_df_sample, in_knowledge, True), 
                             method="least_squares")
    # fitted_params.params.pretty_print()
    
    return fitted_params