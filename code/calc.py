# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 21:28:59 2022

@author: David J. Kedziora
"""

import numpy as np
from scipy.optimize import curve_fit

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

    # Calculate background statistics, especially the average.
    bg_avg = np.mean(bg_samples)
    bg_std = np.std(bg_samples)
    bg_low = np.min(bg_samples)
    bg_high = np.max(bg_samples)
    bg_stats = {"avg": bg_avg,
                "std": bg_std,
                "low": bg_low,
                "high": bg_high}
    
    # Calculate amplitude statistics, correcting for the background.
    # Mask out the MPE peak when processing the other peaks.
    amp_mpe = amps_raw[id_mpe] - bg_avg
    amps_raw.mask[id_mpe] = True
    amp_avg = np.mean(amps_raw) - bg_avg
    amp_std = np.std(amps_raw)
    amp_low = np.min(amps_raw) - bg_avg
    amp_high = np.max(amps_raw) - bg_avg
    amp_stats = {"avg": amp_avg,
                 "std": amp_std,
                 "low": amp_low,
                 "high": amp_high,
                 "mpe": amp_mpe}
    
    # Calculate the 'quality' of the quantum dot.
    g2zero = amp_mpe/amp_avg
    
    return g2zero, amp_stats, bg_stats

# Define the core fitting function for the delay histogram.
# Is based on Eq. (2) of: https://doi.org/10.1063/1.5143786
# x: domain (delays)
# in_knowledge: domain knowledge for how the delay histogram should look
# delay_shift: a minor adjustment for peak delays to get the best fit
# bg: background number of events per bin (noise)
# amp_env: envelope amplitude for the n peaks, which centres at the MPE peak
# amp_ratio: the ratio of the MPE peak to the envelope amplitude, i.e. g2(0)
# inv_decay_env: the inverse decay scale for the envelope
# decay_peak: the decay scale for an individual peak
def func(x, in_knowledge, #*args):
    bg, delay_shift, amp_env, decay_peak, amp_ratio, inv_decay_env):
    # print(args)
    # bg, delay_shift, amp_env, decay_peak, amp_ratio, inv_decay_env = args[0]
    
    # Extract domain knowledge on how the event histogram should look.
    n_peaks = in_knowledge["n_peaks"]
    id_mpe = in_knowledge["id_mpe"]
    pulse_period = in_knowledge["pulse_period"]
    delay_start = in_knowledge["delay_start"]
    
    delay_start += delay_shift  # Adjust the domain knowledge with the fit.
    delay_mpe = delay_start + id_mpe*pulse_period
    # inv_decay_env = 0
    # decay_peak = 1e-9
    
    fit = bg
    for i in range(n_peaks):
        term = amp_env*np.exp(-abs(x-delay_mpe)*inv_decay_env
                              -abs(x-delay_start-i*pulse_period)/decay_peak)
        if i == id_mpe:
            term *= amp_ratio
        fit += term
    return fit

# A basic form of the core fitting function with four arguments.
# The third and fourth denote a common peak amplitude and decay rate.
def func_basic(x, in_knowledge, *args):
    print(args)
    # bg, delay_shift, amp_env, decay_peak = args[0]
    # print("BG %f. Delta %f. Amp all %f. Tau all %f." % (bg, delta, amp_all, tau_all))
    return func(x, in_knowledge, *args[0], 0.5, 0)

# # A fine form of the core fitting function with 2n arguments.
# # Assumes the background and delta are already optimised.
# # The first n args are amplitudes, and the second n args are decay rates.
def func_fine(x, in_knowledge, bg, delay_shift, *args):
    print(args)
    return func(x, in_knowledge, bg, delay_shift, *args[0])

def calc_g2zero_fit(in_df_sample, in_df_delays, in_knowledge):
    
    # Set up initial guesses and bounds for fitting parameters in func above.
    # They are: bg, delay_shift, amp_env, decay_peak, amp_ratio, inv_decay_env
    # n_peaks = in_knowledge["n_peaks"]
    # pulse_period = in_knowledge["pulse_period"]
    # amp_init = 2*sum(in_df_sample)/len(in_df_delays)
    # decay_init = in_df_delays[1] - in_df_delays[0]
    # p_init = [0, 0, amp_init, 0.5, 0, decay_init]
    # p_bounds = ([-pulse_period/2, 0, 0, 0, 0, 0], 
    #             [pulse_period/2, np.inf, np.inf, 1, np.inf, np.inf])
    
    pulse_period = in_knowledge["pulse_period"]
    decay_init = in_df_delays[1] - in_df_delays[0]
    p_init = [0, 0, 1, decay_init]
    p_bounds = ([0, -pulse_period/2, 0, 0],
                [np.inf, pulse_period/2, np.inf, pulse_period])
    
    p_opt, p_cov = curve_fit(lambda x, *p: func_basic(x, in_knowledge, p), xdata=in_df_delays, ydata=in_df_sample, p0=p_init, bounds=p_bounds)
    p_store = list(p_opt[:2])
    p_init = list(p_opt[2:]) + [0.5, 0]
    p_bounds = (p_bounds[0][2:] + [0, 0],
                p_bounds[1][2:] + [1, np.inf])
    p_opt, p_cov = curve_fit(lambda x, *p: func_fine(x, in_knowledge, p_store[0], p_store[1], p), xdata=in_df_delays, ydata=in_df_sample, p0=p_init, bounds=p_bounds)
    p_opt = p_store + list(p_opt)
    print(p_opt)
    
    return p_opt