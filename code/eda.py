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

import calc

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
smoothing_bound = 1e-9     # Unit: s.

# Define experimental device parameters, i.e. for the laser and detectors.
# These will not be used in the fitting methods of deriving g2(0).
pulse_freq = 80e6            # Unit: Hz.
delta_zero = 60.7e-9        # Unit: s.

# Define detection-sampling parameters.
snapshot_bin_size = 10

# SI conversion factors when working with data and later plotting.
delay_unit = 1e-9           # 1 nanosecond.


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
                df_delays = df[df.columns[0]]*delay_unit
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
    # plt.close(fig_hist_snapshot)
    
    
    
    # Sample snapshots of the detection events.
    df_sample_full = df_events.sum(axis=1)
    
    # Rolling-average out the noise for a delay-based histogram of the sample.
    kernel_size = max(1, int(smoothing_bound*2/(delay_bin_size)))
    kernel = np.ones(kernel_size)/kernel_size
    df_sample_full_smooth = np.convolve(df_sample_full, kernel, mode="same")
    
    # Plot the delay-based histogram of the sample, raw and smoothed.
    fig_sample, ax_sample = plt.subplots()
    ax_sample.plot(df_delays/delay_unit, df_sample_full, label="Raw")
    ax_sample.plot(df_delays/delay_unit, df_sample_full_smooth, 
                       label="Rolling Avg. (" + str(kernel_size) + " bins)")
    ax_sample.set_xlabel("Delay (ns)")
    ax_sample.set_ylabel("Events per bin (" + str(delay_bin_size/delay_unit) + " ns)")
    ax_sample.legend()
    fig_sample.savefig(folder_results + filename_prefix + "_hist.png",
                       bbox_inches="tight")
    # plt.close(fig_sample)

    domain_knowledge = calc.compile_domain_knowledge(pulse_freq, 
                                                     delta_zero, delay_range)
    
    g2zero, amp_stats, bg_stats = calc.calc_g2zero_quick(df_sample_full, 
                                                         delay_bin_size, 
                                                         domain_knowledge)
    g2zero_s, amp_stats_s, bg_stats_s = calc.calc_g2zero_quick(df_sample_full_smooth, 
                                                               delay_bin_size, 
                                                               domain_knowledge)