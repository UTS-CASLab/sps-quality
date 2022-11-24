# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:52:15 2022

@author: David J. Kedziora
"""

import matplotlib.pyplot as plt
import numpy as np

import numpy.matlib

def plot_event_history(in_df_events, in_axis_time, in_save_prefix):
    # Create a histogram of events across snapshots and plot it.
    df_hist_snapshot = in_df_events.sum(axis=0)
    
    fig_hist_snapshot, ax_hist_snapshot = plt.subplots()
    ax_hist_snapshot.plot(in_axis_time, df_hist_snapshot, label="Raw")
    ax_hist_snapshot.set_xlabel("Snapshot (s)")
    ax_hist_snapshot.set_ylabel("Events per bin (" 
                                + str(in_axis_time[1] - in_axis_time[0]) + " s)")
    ax_hist_snapshot.legend()
    fig_hist_snapshot.savefig(in_save_prefix + "_events_per_snapshot.png", 
                              bbox_inches="tight")
    plt.close(fig_hist_snapshot)


def plot_event_histogram(in_hist, in_axis_delays, in_delay_unit, in_save_prefix,
                         in_hist_comp = None, in_label_comp = None,
                         in_closeup_xlim = None):
    # Plot a delay-based histogram.
    # Optionally compare it against another function on the same axis.
    fig_sample, ax_sample = plt.subplots()
    ax_sample.plot(in_axis_delays/in_delay_unit, in_hist, label="Raw")
    if (in_hist_comp is not None) and (in_label_comp is not None):
        ax_sample.plot(in_axis_delays/in_delay_unit, in_hist_comp,
                       label=in_label_comp)
    ax_sample.set_xlabel("Delay (%ss)" % in_delay_unit)
    events_per_bin = (in_axis_delays[1]-in_axis_delays[0])/in_delay_unit
    ax_sample.set_ylabel("Events per bin (%s %ss)" % (events_per_bin, in_delay_unit))
    ax_sample.legend()
    fig_sample.savefig(in_save_prefix + "_hist.png",
                       bbox_inches="tight")
    plt.close(fig_sample)
    
    if in_closeup_xlim is not None:
        ax_sample.set_xlim(np.array(in_closeup_xlim)/in_delay_unit)
        fig_sample.savefig(in_save_prefix + "_hist_closeup.png",
                           bbox_inches="tight")
        plt.close(fig_sample)


# TODO: Standardise code to work with dataframes or numpy arrays.
def plot_traces(in_trace_matrix, in_axis_time, 
                in_save_prefix, in_var_name, in_save_suffix = None,
                in_extra_metrics = None, in_extra_colors = None, in_extra_labels = None,
                in_ylim = None):
    
    # Extra metrics must be a list of sub-lists.
    # Each sub-list of values is drawn as horizontal lines on the trace plot.
    # The sub-list itself should have 1, 3 or 5 values.
    # The middle value is the mean and the adjacent values are one std away.
    # The outer values are the lowest and highest values of the metric.
    # Extra colors and extra labels must have one item per each sub-list.
    
    # Input trace matrix should have a row for each trajectory.
    # This code transposes it.
    fig_traces, ax_traces = plt.subplots()
    
    colors = plt.cm.viridis(np.linspace(0, 1, in_trace_matrix.shape[0]))
    
    ax_traces.set_prop_cycle("color", colors)
    ax_traces.plot(in_axis_time, np.transpose(np.array(in_trace_matrix)))
    xlim = [in_axis_time[0], in_axis_time[-1]]
    
    # Plot the extra metrics on top of the traces.
    if in_extra_metrics is not None:
        c = 0
        for metric_spread in in_extra_metrics:
            num_vals = len(metric_spread)
            id_mean = (num_vals - 1)/2
            label_metric = in_extra_labels[c]
            if num_vals > 1:
                id_std_low = id_mean - 1
                id_std_high = id_mean + 1
                ax_traces.plot([xlim, xlim],
                               [[metric_spread[id_std_low], metric_spread[id_std_low]],
                                [metric_spread[id_std_high], metric_spread[id_std_high]]],
                               color = in_extra_colors[c], linestyle = "--")
                label_metric += " + std"
            if num_vals > 3:
                ax_traces.plot([xlim, xlim],
                               [[metric_spread[0], metric_spread[0]],
                                [metric_spread[-1], metric_spread[-1]]],
                               color = in_extra_colors[c], linestyle = ":")
                label_metric += " + extrema"
            ax_traces.plot(xlim, 
                           [metric_spread[id_mean], metric_spread[id_mean]], 
                           color = in_extra_colors[c], linestyle = "-",
                           label = in_extra_labels[c])
            c += 1
            
    ax_traces.set_xlabel("Total Detection Time (s)")
    ax_traces.set_ylabel(in_var_name)
    ax_traces.set_xlim(xlim)
    if not in_ylim is None:
        ax_traces.set_ylim(in_ylim)
    savefile = in_save_prefix + "_trace_" + in_var_name
    if not in_save_suffix is None:
        savefile += "_" + in_save_suffix
    savefile += ".png"
    if in_extra_labels is not None:
        ax_traces.legend()
    fig_traces.savefig(savefile, bbox_inches="tight")
    plt.close(fig_traces)