# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 21:28:59 2022

@author: David J. Kedziora
"""

import numpy as np

def calc_domain_knowledge(pulse_freq, pulse_freq_unit, pulse_period_unit):
    
    # From pulse details, calculate expected number of peaks in the histogram.
    # Work out which peak is the zero-delay multi-photon-emission (MPE) peak.
    # Work out at which raw delay the peaks start.
    knowledge["pulse_period"] = 1/(pulse_freq*pulse_freq_unit*pulse_period_unit)
    knowledge["n_peaks"] = 1 + int(delay_range/pulse_period)
    knowledge["id_mpe"] = int(delta_zero/pulse_period)
    knowledge["delay_start"] = np.mod(delta_zero, pulse_period)
    

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
        search_min = max(round((delay_start - pulse_period/4 + i*pulse_period)
                               /in_bin_size), 
                         0)
        search_max = min(round((delay_start + pulse_period/4 + i*pulse_period)
                               /in_bin_size), 
                         in_df_sample.size-1)
        amps_raw[i] = max(in_df_sample[search_min:search_max])
        if i != 0:
            id_bg_sample = round((delay_start + (i-1/2)*pulse_period)/in_bin_size)
            bg_samples[i-1] = in_df_sample[id_bg_sample]

    # Calculate background statistics, especially the average.
    bg_avg = np.mean(bg_samples)
    bg_std = np.std(bg_samples)
    bg_low = min(bg_samples)
    bg_high = max(bg_samples)
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
    amp_low = min(amps_raw) - bg_avg
    amp_high = max(amps_raw) - bg_avg
    amp_stats = {"avg": amp_avg,
                 "std": amp_std,
                 "low": amp_low,
                 "high": amp_high}
    
    # Calculate the 'quality' of the quantum dot.
    g2zero = amp_mpe/amp_avg
    
    return g2zero, amp_stats, bg_stats