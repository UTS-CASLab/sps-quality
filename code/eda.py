# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:53:41 2022

@author: David J. Kedziora
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time



filename_prefixes = ["1p2uW_3000cps_time bin width 128 ps",
                     "2p5uW_4000cps",
                     "4uW_4100cps",
                     "8uW_5100cps",
                     "10uW_6000cps",
                     "10uW_12000cps",
                     "20uW_7000cps",
                     "30uW_7000cps"]
folder_data = "../data/"
folder_results = "../results/"

# Define how far the bound of several windows should be from its centre.
# These are used for the quick methods of deriving g2(0).
# E.g. +-3 ns at 0.256 ns per delay bin means 23 bins are used in the window.
smoothing_bound = 1
smoothing_unit = 1e-9     # 1 nanosecond.

# Define experimental device parameters, i.e. for the laser and detectors.
# These will not be used in the fitting methods of deriving g2(0).
pulse_freq = 80
pulse_freq_unit = 1e6       # 1 MHz.
pulse_period = None         # Easily calculated later from the variables above.
pulse_period_unit = 1e-9    # 1 nanosecond.
delta_zero = 60.7
delta_unit = 1e-9           # 1 nanosecond.

# Define detection-sampling parameters.
snapshot_bin_size = 10
snapshot_unit = 1       # 1 second.
delay_bin_size = None   # Easily calculated later from the data.
delay_unit = 1e-9       # 1 nanosecond.


for filename_prefix in filename_prefixes:
    # Loads data files into a single dataframe.
    # Establishes delay/snapshot arrays, as well as a detection event matrix.
    # WARNING: There is no bug checking. Ensure files are appropriate to merge.
    df_delays = None
    df_events = None
    for filename_data in os.listdir(folder_data):
        if filename_data.startswith(filename_prefix):
            df = pd.read_csv(folder_data + filename_data, sep="\t", header=None)
            if df_events is None:
                print("Loading into dataframe: %s" % folder_data + filename_data)
                df_delays = df[df.columns[0]]
                df_events = df[df.columns[1:]]
            else:
                print("Merging into dataframe: %s" % folder_data + filename_data)
                df_events = pd.concat([df_events, df[df.columns[1:]]], axis=1)
    df_snapshots = range(0, 
                         df_events.columns.size * snapshot_bin_size, 
                         snapshot_bin_size)
    df_events.columns = df_snapshots
    
    # Calculate helper variables.
    delay_bin_size = df_delays[1] - df_delays[0]
    delay_range = df_delays[df_delays.size-1] - df_delays[0]
    
    # Create a histogram of events across snapshots and plot it.
    df_hist_snapshot = df_events.sum(axis=0)
    
    fig_hist_snapshot, ax_hist_snapshot = plt.subplots()
    ax_hist_snapshot.plot(df_snapshots, df_hist_snapshot, label="Raw")
    ax_hist_snapshot.set_xlabel("Snapshot (s)")
    ax_hist_snapshot.set_ylabel("Events per bin (" 
                                + str(snapshot_bin_size) + " s)")
    ax_hist_snapshot.legend()
    fig_hist_snapshot.savefig(folder_results + filename_prefix + "_events_per_snapshot.png", 
                              bbox_inches="tight")
    
    
    
    # Sample snapshots of the detection events.
    df_sample = df_events.sum(axis=1)
    
    # Rolling-average out the noise for a delay-based histogram of the sample.
    kernel_size = max(1, int(smoothing_bound*smoothing_unit*2/(delay_bin_size*delay_unit)))
    kernel = np.ones(kernel_size)/kernel_size
    df_sample_smooth = np.convolve(df_sample, kernel, mode="same")
    
    # Plot the delay-based histogram of the sample, raw and smoothed.
    fig_sample, ax_sample = plt.subplots()
    ax_sample.plot(df_delays, df_sample, label="Raw")
    ax_sample.plot(df_delays, df_sample_smooth, 
                       label="Rolling Avg. (" + str(kernel_size) + " bins)")
    ax_sample.set_xlabel("Delay (ns)")
    ax_sample.set_ylabel("Events per bin (" + str(delay_bin_size) + " ns)")
    ax_sample.legend()
    fig_sample.savefig(folder_results + filename_prefix + "_hist_smoothed.png",
                       bbox_inches="tight")
    
    
    pulse_period = 1/(pulse_freq*pulse_freq_unit*pulse_period_unit)
    n_peaks = 1 + int(delay_range/pulse_period)
    id_mpe = int(delta_zero/pulse_period)
    delay_start = np.mod(delta_zero, pulse_period)
    
    amps = np.ma.array(np.zeros(n_peaks), mask=False)
    for i in range(n_peaks):
        search_min = max(round((delay_start - pulse_period/4 + i*pulse_period)
                               /delay_bin_size), 
                         0)
        search_max = min(round((delay_start + pulse_period/4 + i*pulse_period)
                               /delay_bin_size), 
                         df_delays.size-1)
        amps[i] = max(df_sample[search_min:search_max])
    amp_mpe = amps[id_mpe]
    amps.mask[id_mpe] = True
    amp_low = min(amps)
    amp_high = max(amps)
    amp_avg = np.mean(amps)
    amp_std = np.std(amps)
    