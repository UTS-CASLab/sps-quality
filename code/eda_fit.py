# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:30:41 2023

@author: David J. Kedziora
"""

import os

import numpy as np

import load
import calc
import plot

folder_data = "../data/"
folder_plots = "../results/"
folder_saves = "../saves/"

# Only full filename prefixes that contain a listed substring will be loaded.
full_filename_requirements = ["10uW"]

random_seed = 0

constants = {}
constants["duration_snapshot"] = 10     # Single sampling by detectors; unit s.
constants["unit_delay"] = 1e-9          # SI unit for delays; 1 ns.

full_filename_prefixes = load.get_full_filename_prefixes(folder_data)

for full_filename_prefix in full_filename_prefixes:
    
    # Do not load filenames that do not contain a required substring.
    if not ((full_filename_requirements is None) or (len(full_filename_requirements) == 0)):
        is_filename_skippable = True
        
        for requirement in full_filename_requirements:
            if requirement in full_filename_prefix:
                is_filename_skippable = False
                
        if is_filename_skippable:
            continue
    
    # Load the experiment.
    df_events, sr_delays, range_snapshots = load.load_experiment(folder_data, full_filename_prefix, constants)
    
    # Prepares save destination for any plots.
    plot_prefix = folder_plots + full_filename_prefix
    
    folder_prefix, filename_prefix = os.path.split(full_filename_prefix)
    for folder_base in [folder_plots, folder_saves]:
        if not os.path.exists(os.path.join(folder_base, folder_prefix)):
            os.makedirs(os.path.join(folder_base, folder_prefix))
            
    xlim_closeup = [60e-9 - (1/80e6)*3/2,
                    60e-9 + (1/80e6)*3/2]
    
    order_snapshot = np.random.permutation(range_snapshots)
    
    sr_sample = df_events.iloc[:,0:-1].sum(axis=1)
    sr_sample /= np.sum(sr_sample)  # Normalise into a probability distribution.
    
    fit_result = calc.estimate_g2zero_pulsed(sr_sample, sr_delays)
    fit_result.params.pretty_print()
    
    # Plot the delay-based histogram of the sample, raw and fitted.
    plot.plot_event_histogram(sr_sample, sr_delays, constants["unit_delay"], plot_prefix + "_fit",
                              in_hist_comp = calc.func_pulsed(fit_result.params, sr_delays, sr_sample), 
                              in_label_comp = "Fit",
                              in_closeup_xlim = xlim_closeup)