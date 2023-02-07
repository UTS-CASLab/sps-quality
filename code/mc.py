# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 21:56:24 2023

@author: David J. Kedziora
"""

import os

import pandas as pd
import numpy as np
from lmfit import fit_report

from time import time

import load
import calc
import plot

import matplotlib.pyplot as plt
from scipy import optimize as sco

folder_data = "../data/"
folder_plots = "../results/"
folder_saves = "../saves/"

# Only full filename prefixes that contain a listed substring will be loaded.
full_filename_requirements = ["SEQUR"]
# full_filename_requirements = ["10uW"]
# full_filename_requirements = ["1p2uW"]

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
            
    # Fit the full sample for the 'best' parameters.
    sr_best = df_events.sum(axis=1)
    print("This dataset details %i two-photon events over %i seconds." 
          % (sr_best.sum(), len(range_snapshots)*constants["duration_snapshot"]))
    sr_best /= sr_best.sum()
    
    # fit_result_poisson = calc.estimate_g2zero_pulsed(sr_best, sr_delays, knowns, use_poisson_likelihood = True)
    fit_result = calc.estimate_g2zero_pulsed(sr_best, sr_delays, knowns, use_poisson_likelihood = False)
                    
    sr_fit = calc.func_pulsed(fit_result.params, sr_delays)
    plot.plot_event_histogram(sr_best, sr_delays, constants["unit_delay"], 
                              plot_prefix + "_test",
                              in_hist_comp = sr_fit, 
                              in_label_comp = "Fit",
                              in_xlim_closeup = xlim_closeup)
    # TODO: Actually log this. Its display is currently almost useless.
    print(fit_report(fit_result))
    
    # Plot the integral of the histogram function with 'best' parameters.
    d_delays = sr_delays[1]-sr_delays[0]
    n_delays = len(sr_delays)
    
    sr_fit_int = calc.func_pulsed_integral(fit_result.params, sr_delays[0], sr_delays)
    
    fig_int, ax_int = plt.subplots()
    ax_int.plot(sr_delays, sr_fit_int, label="~CDF")
    ax_int.set_xlabel("Delay (%ss)" % constants["unit_delay"])
    ax_int.set_ylabel("Integral")
    ax_int.legend()
    fig_int.savefig(plot_prefix + "_test_hist_integrated.png", bbox_inches="tight")
    
    plt.close(fig_int)
    
    # Recall that the series of delays marks the lower bounds of histogram bins.
    # The total integral of the continuous distribution must take the highest upper bound.
    total_int = calc.func_pulsed_integral(fit_result.params, sr_delays[0], n_delays*d_delays)
    
    # Define a function where the root indicates the following...
    # The x value of the histogram integral corresponding to a specified y value.
    def cdf_root(x, y, in_params, in_domain_start):
        return calc.func_pulsed_integral(in_params, in_domain_start, x) - y
    
    def cdf_derivative(x, y, in_params, in_domain_start):
        return calc.func_pulsed(in_params, x)
    
    np.random.seed(seed = random_seed)
    
    sr_sample = pd.Series(np.zeros(n_delays))
    
    t = time()
    num_events = 1000
    
    for k in range(num_events):
    
        y_sample = np.random.uniform(low=0.0, high=total_int)
        x_guess = (y_sample/total_int)*(n_delays*d_delays-sr_delays[0])
        
        root_result = sco.root_scalar(cdf_root, args=(y_sample, fit_result.params, sr_delays[0]),
                                      bracket = [sr_delays[0], n_delays*d_delays],
                                      x0 = x_guess)
        
        sr_sample[np.floor_divide(root_result.root, d_delays)] += 1
        
    print("%i two-photon events inverse-sampled from integral: %f s" % (num_events, time() - t))
        
    sr_sample /= sr_sample.sum()
    
    plot.plot_event_histogram(sr_sample, sr_delays, constants["unit_delay"], 
                              plot_prefix + "_mc",
                              in_hist_comp = sr_fit, 
                              in_label_comp = "Sampled Fit",
                              in_xlim_closeup = xlim_closeup)