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
import multiprocessing as mp
from functools import partial

import load
import calc
import plot

import matplotlib.pyplot as plt

def main():
    
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
    
    # Set a hardware dependent sample size above which sampling uses multiprocessing.
    # This avoids multiprocessing overhead on small samplings.
    sample_size_threshold_single_process = 3000
    print("CPU Count: %i" % mp.cpu_count())
    
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
        
        use_poisson_likelihood = False
        
        # fit_result_poisson = calc.estimate_g2zero_pulsed(sr_best, sr_delays, knowns, use_poisson_likelihood = True)
        fit_result = calc.estimate_g2zero_pulsed(sr_best, sr_delays, knowns, use_poisson_likelihood)
                        
        sr_fit = calc.func_pulsed(fit_result.params, sr_delays)
        plot.plot_event_histogram(sr_best, sr_delays, constants["unit_delay"], 
                                  plot_prefix + "_test",
                                  in_hist_comp = sr_fit, 
                                  in_label_comp = "Fit",
                                  in_xlim_closeup = xlim_closeup)
        # TODO: Actually log this. Its display is currently almost useless.
        print(fit_report(fit_result))
        
        # Extract details of the delay domain and note bin edges.
        d_delays = sr_delays[1] - sr_delays[0]
        n_delays = len(sr_delays)
        sr_edges = pd.concat([sr_delays, pd.Series([n_delays*d_delays])], ignore_index=True)
        
        # Plot the integral of the histogram function with 'best' parameters.
        sr_fit_int = calc.func_pulsed_integral(fit_result.params, sr_edges[0], sr_edges)
        
        fig_int, ax_int = plt.subplots()
        ax_int.plot(sr_edges, sr_fit_int, label="~CDF")
        ax_int.set_xlabel("Delay (%ss)" % constants["unit_delay"])
        ax_int.set_ylabel("Integral")
        ax_int.legend()
        fig_int.savefig(plot_prefix + "_test_hist_integrated.png", bbox_inches="tight")
        
        plt.close(fig_int)
        
        # Recall that the series of delays marks the lower bounds of histogram bins.
        # The total integral of the continuous distribution must take the highest upper bound.
        total_int = calc.func_pulsed_integral(fit_result.params, sr_edges[0], sr_edges.iloc[-1])
        
        # Generate samples
        for repeat in range(1):
            for size_exponent in range(2, 7):
                num_events = 10**size_exponent
                
                t = time()
                
                # Uniformly sample possible integral values within the delay domain.
                np.random.seed(seed = random_seed)
                y_sample = np.random.uniform(low=0.0, high=total_int, size=num_events)
                
                results = np.zeros(num_events)
                label_process = "single"
                # Use multiprocessing if generating a lot of data.
                if num_events > sample_size_threshold_single_process:
                    label_process = "multi"
                    with mp.Pool(processes = (mp.cpu_count() - 1)) as pool:
                        results = pool.map(partial(calc.inverse_sample, 
                                                    in_params = fit_result.params, 
                                                    in_domain_start = sr_edges[0],
                                                    in_domain_end = sr_edges[0] + sr_edges.iloc[-1],
                                                    in_total_int = total_int), 
                                            y_sample)
                
                else:
                    for k in range(num_events):
                        y_random = y_sample[k]
                        
                        results[k] = calc.inverse_sample(y_random, 
                                                         in_params = fit_result.params, 
                                                         in_domain_start = sr_edges[0],
                                                         in_domain_end = sr_edges[0] + sr_edges.iloc[-1],
                                                         in_total_int = total_int)
                
                # The binned results should have a length one less than the number of bin edges.
                array_sample = np.histogram(results, bins=sr_edges)[0].astype(float)
                array_sample /= np.sum(array_sample)
                
                # Include fitting procedure in the timing.
                # TODO: Expand this out.
                fit_sample = calc.estimate_g2zero_pulsed(array_sample, sr_delays, knowns, use_poisson_likelihood)
                    
                print("%i two-photon events inverse-sampled from integral (%s-process): %f s" 
                      % (num_events, label_process, time() - t))
                
                plot.plot_event_histogram(array_sample, sr_delays, constants["unit_delay"], 
                                          plot_prefix + "_mc_" + str(num_events),
                                          in_hist_comp = sr_fit, 
                                          in_label_comp = "Sampled Fit",
                                          in_xlim_closeup = xlim_closeup)
        
if __name__ == '__main__':
    main()