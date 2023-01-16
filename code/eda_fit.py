# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:30:41 2023

@author: David J. Kedziora
"""

import os

import pandas as pd
import numpy as np

from time import time

import load
import calc
import plot

folder_data = "../data/"
folder_plots = "../results/"
folder_saves = "../saves/"

# Only full filename prefixes that contain a listed substring will be loaded.
full_filename_requirements = ["SEQUR"]
# full_filename_requirements = ["10uW"]

random_seed = 0

constants = {}
constants["duration_snapshot"] = 10     # Single sampling by detectors; unit s.
constants["unit_delay"] = 1e-9          # SI unit for delays; 1 ns.

knowns = {}
knowns["period_pulse"] = 1/80e6         # Inverse of laser frequency in Hz.
knowns["delay_mpe"] = [55e-9, 65e-9]    # Delay range where multi-photon events occur.

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
    
    # Prepare save destination for any plots and results.
    plot_prefix = folder_plots + full_filename_prefix
    save_prefix = folder_saves + full_filename_prefix + "_seed_" + str(random_seed)
    
    folder_prefix, filename_prefix = os.path.split(full_filename_prefix)
    for folder_base in [folder_plots, folder_saves]:
        if not os.path.exists(os.path.join(folder_base, folder_prefix)):
            os.makedirs(os.path.join(folder_base, folder_prefix))
    
    # Set up a zoom for histogram plots.
    xlim_closeup = [np.mean(knowns["delay_mpe"]) - knowns["period_pulse"]*3/2,
                    np.mean(knowns["delay_mpe"]) + knowns["period_pulse"]*3/2]
    
    # Determine fitting parameters of interest.
    param_ids = ["amp_env", "amp_ratio", "bg", "decay_peak", "delay_mpe"]
    param_fits = dict()
    
    # Load fit results from pickled save files if they exist.
    do_generate_fits = False
    try:
        for param_id in param_ids:
            param_fits[param_id] = pd.read_pickle(save_prefix + "_fits_" + param_id + ".pkl")
        
        print("Previously saved fits for seed %i loaded." % random_seed)
    except:
        do_generate_fits = True
    
    # If fits have not been done before, do them now.
    if do_generate_fits:
        print("Generating fits for seed %i." % random_seed)
        
        # Shuffle the order of detector snapshots.
        # TODO: Watch out for snapshots where no detections occurred.
        order_snapshot = np.random.permutation(range_snapshots)
        df_events_shuffled = df_events[order_snapshot]
        # df_events_shuffled = df_events_shuffled.iloc[:,0:50]
        max_snapshots = len(df_events_shuffled.columns)
        
        # Determine the number of samples of different sizes that should be fitted.
        sizes_sample = np.array([2**i for i in range(0, int(np.log2(max_snapshots)))] + [max_snapshots])
        # sizes_sample = np.array([max_snapshots])
        number_fits = np.sum((max_snapshots/sizes_sample).astype(int))
        
        # Create dataframes to store the fit results.
        param_fit_ids = ["size", "id", "value", "stderr"]
        for param_id in param_ids:
            param_fits[param_id] = pd.DataFrame(np.zeros([number_fits, len(param_fit_ids)]))
            param_fits[param_id].columns = param_fit_ids
        
        # Do the fits for various sample sizes and store the results.
        count_fit = 0
        for size_sample in sizes_sample:
            number_samples = int(max_snapshots/size_sample)
            t = time()
            for i in range(number_samples):
                # Sum the samples and normalise.
                sr_sample = df_events_shuffled.iloc[:, size_sample*i:size_sample*(i+1)].sum(axis=1)
                sr_sample /= sr_sample.sum()
                
                fit_result = calc.estimate_g2zero_pulsed(sr_sample, sr_delays, knowns)
                
                for param_id in param_ids:
                    param_fits[param_id]["size"][count_fit] = size_sample
                    param_fits[param_id]["id"][count_fit] = i
                    param_fits[param_id]["value"][count_fit] = fit_result.params[param_id].value
                    param_fits[param_id]["stderr"][count_fit] = fit_result.params[param_id].stderr
                count_fit += 1
            print("%i samples of size %i (max %i) have been fitted: %f s" 
                  % (number_samples, size_sample, max_snapshots, time() - t))
            
            # As a sanity check, compare fit and histogram on the last sample.
            plot.plot_event_histogram(sr_sample, sr_delays, constants["unit_delay"], 
                                      plot_prefix + "_example_fit_sample_size_" + str(size_sample),
                                      in_hist_comp = calc.func_pulsed(fit_result.params, sr_delays, sr_sample), 
                                      in_label_comp = "Fit",
                                      in_xlim_closeup = xlim_closeup)
        
        # Save the results in pickled format.
        for param_id in param_ids:
            param_fits[param_id].to_pickle(save_prefix + "_fits_" + param_id + ".pkl")
            
    for param_id in param_ids:
        plot.plot_param_fits(param_fits[param_id], plot_prefix + "_" + param_id)
    
    # sr_sample = df_events_shuffled.iloc[:,0:-1].sum(axis=1)
    # sr_sample /= np.sum(sr_sample)  # Normalise into a probability distribution.
    
    # t = time()
    # fit_result = calc.estimate_g2zero_pulsed(sr_sample, sr_delays, knowns)
    # fit_result.params.pretty_print()
    # print("Fit for %i parameters: %f s" % (len(fit_result.params), time() - t))
    
    # # Plot the delay-based histogram of the sample, raw and fitted.
    # plot.plot_event_histogram(sr_sample, sr_delays, constants["unit_delay"], plot_prefix + "_fit",
    #                           in_hist_comp = calc.func_pulsed(fit_result.params, sr_delays, sr_sample), 
    #                           in_label_comp = "Fit",
    #                           in_xlim_closeup = xlim_closeup)