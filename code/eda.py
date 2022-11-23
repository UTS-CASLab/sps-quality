# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:53:41 2022

@author: David J. Kedziora
"""

import os
import pandas as pd
import numpy as np

from time import time

import calc
import plot

filename_prefixes = [
    # "1p2uW_3000cps_time bin width 128 ps",
    # "2p5uW_4000cps",
    # "4uW_4100cps",
    # "8uW_5100cps",
    # "10uW_6000cps",
    # "10uW_12000cps",
    # "20uW_7000cps",
    "30uW_7000cps"
    ]
folder_data = "../data/"
folder_plots = "../results/"
folder_saves = "../saves/"
random_seed = 0

# The number of traces to generate per datafile.
# Each trace shows how g2(0) estimates change with greater sampling.
num_traces = 3

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
    
    # Prepares save destination for any plots and storable results.
    plot_prefix = folder_plots + filename_prefix
    save_prefix = folder_saves + filename_prefix + "_traces_" + str(num_traces) + "_seed_" + str(random_seed)
    
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
    range_snapshots = range(0, df_events.columns.size * snapshot_bin_size, 
                            snapshot_bin_size)
    df_events.columns = range_snapshots
    
    # Calculate helper variables; these may be recalculated inside functions.
    delay_bin_size = df_delays[1] - df_delays[0]
    delay_range = df_delays[df_delays.size-1] - df_delays[0]
    
    # Create a histogram of events across snapshots and plot it.
    plot.plot_event_history(df_events, range_snapshots, plot_prefix)
    
    # Sum all snapshots of the detection events.
    df_sample_full = df_events.sum(axis=1)
    
    # Generate domain knowledge on how the histogram should look like.
    domain_knowledge = calc.compile_domain_knowledge(pulse_freq, 
                                                     delta_zero, delay_range)
    closeup_xlim = [delta_zero - domain_knowledge["pulse_period"]*3/2,
                    delta_zero + domain_knowledge["pulse_period"]*3/2]
     
    
    # Rolling-average out the noise for a delay-based histogram of the sample.
    kernel_size = max(1, int(smoothing_bound*2/(delay_bin_size)))
    kernel = np.ones(kernel_size)/kernel_size
    df_sample_full_smooth = np.convolve(df_sample_full, kernel, mode="same")
    
    # Plot the delay-based histogram of the full sample, raw and smoothed.
    plot.plot_event_histogram(df_sample_full, df_delays, delay_unit, plot_prefix + "_smooth",
                              in_hist_comp = df_sample_full_smooth, 
                              in_label_comp = "Rolling Avg. (" + str(kernel_size) + " bins)",
                              in_closeup_xlim = closeup_xlim)
    
    # Fit the expectation of the histogram: https://doi.org/10.1063/1.5143786
    t = time()
    fit_result = calc.calc_g2zero_fit(df_sample_full, df_delays, domain_knowledge)
    print("Fit for %i parameters: %f s" % (len(fit_result.params), time() - t))
    print("g2(0) = %f +- %f" % (fit_result.params["amp_ratio"].value, fit_result.params["amp_ratio"].stderr))
    
    # Plot the delay-based histogram of the full sample, raw and fitted.
    plot.plot_event_histogram(df_sample_full, df_delays, delay_unit, plot_prefix + "_fit",
                              in_hist_comp = calc.func(fit_result.params, df_delays, df_sample_full, domain_knowledge), 
                              in_label_comp = "Fit",
                              in_closeup_xlim = closeup_xlim)
    
    # Load trace data from pickle save files if they exist.
    do_generate_traces = False
    try:
        df_trace_g2zero = pd.read_pickle(save_prefix + "_g2zero.pkl")
        df_trace_amp_mpe = pd.read_pickle(save_prefix + "_amp_mpe.pkl")
        df_trace_amp_avg = pd.read_pickle(save_prefix + "_amp_avg.pkl")
        df_trace_amp_std = pd.read_pickle(save_prefix + "_amp_std.pkl")
        df_trace_bg_avg = pd.read_pickle(save_prefix + "_bg_avg.pkl")
        df_trace_bg_std = pd.read_pickle(save_prefix + "_bg_std.pkl")
        
        df_trace_g2zero_s = pd.read_pickle(save_prefix + "_g2zero_smooth.pkl")
        df_trace_amp_mpe_s = pd.read_pickle(save_prefix + "_amp_mpe_smooth.pkl")
        df_trace_amp_avg_s = pd.read_pickle(save_prefix + "_amp_avg_smooth.pkl")
        df_trace_amp_std_s = pd.read_pickle(save_prefix + "_amp_std_smooth.pkl")
        df_trace_bg_avg_s = pd.read_pickle(save_prefix + "_bg_avg_smooth.pkl")
        df_trace_bg_std_s = pd.read_pickle(save_prefix + "_bg_std_smooth.pkl")
        
        print("%i previously saved traces for seed %i loaded." 
          % (num_traces, random_seed))
    except:
        do_generate_traces = True
        print("Generating %i traces for seed %i." % (num_traces, random_seed))
    
    
    # Generate traces of how quality estimates change over time.
    if do_generate_traces:
    
    
        np.random.seed(random_seed)
        df_trace_g2zero = pd.DataFrame(np.zeros([num_traces, len(range_snapshots)]))
        df_trace_amp_mpe = pd.DataFrame(np.zeros([num_traces, len(range_snapshots)]))
        df_trace_amp_avg = pd.DataFrame(np.zeros([num_traces, len(range_snapshots)]))
        df_trace_amp_std = pd.DataFrame(np.zeros([num_traces, len(range_snapshots)]))
        df_trace_bg_avg = pd.DataFrame(np.zeros([num_traces, len(range_snapshots)]))
        df_trace_bg_std = pd.DataFrame(np.zeros([num_traces, len(range_snapshots)]))
        
        df_trace_g2zero_s = pd.DataFrame(np.zeros([num_traces, len(range_snapshots)]))
        df_trace_amp_mpe_s = pd.DataFrame(np.zeros([num_traces, len(range_snapshots)]))
        df_trace_amp_avg_s = pd.DataFrame(np.zeros([num_traces, len(range_snapshots)]))
        df_trace_amp_std_s = pd.DataFrame(np.zeros([num_traces, len(range_snapshots)]))
        df_trace_bg_avg_s = pd.DataFrame(np.zeros([num_traces, len(range_snapshots)]))
        df_trace_bg_std_s = pd.DataFrame(np.zeros([num_traces, len(range_snapshots)]))
        
        t = time()
        for i in range(num_traces):
            order_snapshot = np.random.permutation(range_snapshots)
            hist = np.zeros(df_delays.size)
            
            count_snapshot = 0
            for id_snapshot in order_snapshot:
                hist = hist + df_events[id_snapshot]
                hist_smooth = np.convolve(hist, kernel, mode="same")
        
                g2zero, amp_stats, bg_stats = calc.calc_g2zero_quick(hist,
                                                                      delay_bin_size,
                                                                      domain_knowledge)
                g2zero_s, amp_stats_s, bg_stats_s = calc.calc_g2zero_quick(hist_smooth,
                                                                            delay_bin_size,
                                                                            domain_knowledge)
                
                df_trace_g2zero.iloc[i, count_snapshot] = g2zero
                df_trace_amp_mpe.iloc[i, count_snapshot] = amp_stats["mpe"]
                df_trace_amp_avg.iloc[i, count_snapshot] = amp_stats["avg"]
                df_trace_amp_std.iloc[i, count_snapshot] = amp_stats["std"]
                df_trace_bg_avg.iloc[i, count_snapshot] = bg_stats["avg"]
                df_trace_bg_std.iloc[i, count_snapshot] = bg_stats["std"]
    
                df_trace_g2zero_s.iloc[i, count_snapshot] = g2zero_s
                df_trace_amp_mpe_s.iloc[i, count_snapshot] = amp_stats_s["mpe"]
                df_trace_amp_avg_s.iloc[i, count_snapshot] = amp_stats_s["avg"]
                df_trace_amp_std_s.iloc[i, count_snapshot] = amp_stats_s["std"]
                df_trace_bg_avg_s.iloc[i, count_snapshot] = bg_stats_s["avg"]
                df_trace_bg_std_s.iloc[i, count_snapshot] = bg_stats_s["std"]
                
                count_snapshot += 1
        print("%i estimated g2(0) trajectories across %i snapshots: %f s" 
              % (num_traces, len(range_snapshots), time() - t))
        
        # Save trace data to pickle files.
        df_trace_g2zero.to_pickle(save_prefix + "_g2zero.pkl")
        df_trace_amp_mpe.to_pickle(save_prefix + "_amp_mpe.pkl")
        df_trace_amp_avg.to_pickle(save_prefix + "_amp_avg.pkl")
        df_trace_amp_std.to_pickle(save_prefix + "_amp_std.pkl")
        df_trace_bg_avg.to_pickle(save_prefix + "_bg_avg.pkl")
        df_trace_bg_std.to_pickle(save_prefix + "_bg_std.pkl")
        
        df_trace_g2zero_s.to_pickle(save_prefix + "_g2zero_smooth.pkl")
        df_trace_amp_mpe_s.to_pickle(save_prefix + "_amp_mpe_smooth.pkl")
        df_trace_amp_avg_s.to_pickle(save_prefix + "_amp_avg_smooth.pkl")
        df_trace_amp_std_s.to_pickle(save_prefix + "_amp_std_smooth.pkl")
        df_trace_bg_avg_s.to_pickle(save_prefix + "_bg_avg_smooth.pkl")
        df_trace_bg_std_s.to_pickle(save_prefix + "_bg_std_smooth.pkl")
    
    # Calculate relative standard deviations (RSDs).
    df_trace_amp_rsd = df_trace_amp_std/df_trace_amp_avg
    df_trace_bg_rsd = df_trace_bg_std/df_trace_bg_avg
    df_trace_amp_rsd_s = df_trace_amp_std_s/df_trace_amp_avg_s
    df_trace_bg_rsd_s = df_trace_bg_std_s/df_trace_bg_avg_s
    
    # Plot traces for the number of trajectories and random seed chosen.
    plot.plot_traces(df_trace_g2zero, range_snapshots, plot_prefix, "g2(0)",
                      in_ylim = [0, 1])
    plot.plot_traces(df_trace_amp_mpe, range_snapshots, plot_prefix, "amp_mpe",
                      in_ylim = [0, np.max([df_trace_amp_mpe.max().max(), df_trace_amp_mpe_s.max().max()])])
    plot.plot_traces(df_trace_amp_avg, range_snapshots, plot_prefix, "amp_avg",
                      in_ylim = [0, np.max([df_trace_amp_avg.max().max(), df_trace_amp_avg_s.max().max()])])
    plot.plot_traces(df_trace_amp_std, range_snapshots, plot_prefix, "amp_std",
                      in_ylim = [0, np.max([df_trace_amp_std.max().max(), df_trace_amp_std_s.max().max()])])
    plot.plot_traces(df_trace_bg_avg, range_snapshots, plot_prefix, "bg_avg",
                      in_ylim = [0, np.max([df_trace_bg_avg.max().max(), df_trace_bg_avg_s.max().max()])])
    plot.plot_traces(df_trace_bg_std, range_snapshots, plot_prefix, "bg_std",
                      in_ylim = [0, np.max([df_trace_bg_std.max().max(), df_trace_bg_std_s.max().max()])])
    
    plot.plot_traces(df_trace_amp_rsd, range_snapshots, plot_prefix, "amp_rsd",
                      in_ylim = [0, 1])
    plot.plot_traces(df_trace_bg_rsd, range_snapshots, plot_prefix, "bg_rsd",
                      in_ylim = [0, 1])
    
    plot.plot_traces(df_trace_g2zero_s, range_snapshots, plot_prefix, "g2(0)", "smoothed",
                      in_ylim = [0, 1])
    plot.plot_traces(df_trace_amp_mpe_s, range_snapshots, plot_prefix, "amp_mpe", "smoothed",
                      in_ylim = [0, np.max([df_trace_amp_mpe.max().max(), df_trace_amp_mpe_s.max().max()])])
    plot.plot_traces(df_trace_amp_avg_s, range_snapshots, plot_prefix, "amp_avg", "smoothed",
                      in_ylim = [0, np.max([df_trace_amp_avg.max().max(), df_trace_amp_avg_s.max().max()])])
    plot.plot_traces(df_trace_amp_std_s, range_snapshots, plot_prefix, "amp_std", "smoothed",
                      in_ylim = [0, np.max([df_trace_amp_std.max().max(), df_trace_amp_std_s.max().max()])])
    plot.plot_traces(df_trace_bg_avg_s, range_snapshots, plot_prefix, "bg_avg", "smoothed",
                      in_ylim = [0, np.max([df_trace_bg_avg.max().max(), df_trace_bg_avg_s.max().max()])])
    plot.plot_traces(df_trace_bg_std_s, range_snapshots, plot_prefix, "bg_std", "smoothed",
                      in_ylim = [0, np.max([df_trace_bg_std.max().max(), df_trace_bg_std_s.max().max()])])
    
    plot.plot_traces(df_trace_amp_rsd_s, range_snapshots, plot_prefix, "amp_rsd",
                      in_ylim = [0, 1])
    plot.plot_traces(df_trace_bg_rsd_s, range_snapshots, plot_prefix, "bg_rsd",
                      in_ylim = [0, 1])