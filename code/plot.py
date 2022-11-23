# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:52:15 2022

@author: David J. Kedziora
"""

import matplotlib.pyplot as plt
import numpy as np

from copy import copy
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
                in_ylim = None):
    
    # Input trace matrix should have a row for each trajectory.
    # This code transposes it.
    fig_traces, ax_traces = plt.subplots()
    
    colors = plt.cm.viridis(np.linspace(0, 1, in_trace_matrix.shape[0]))
    
    ax_traces.set_prop_cycle("color", colors)
    ax_traces.plot(in_axis_time, np.transpose(np.array(in_trace_matrix)))
    ax_traces.set_xlabel("Total Detection Time (s)")
    ax_traces.set_ylabel(in_var_name)
    ax_traces.set_xlim([in_axis_time[0], in_axis_time[-1]])
    if not in_ylim is None:
        ax_traces.set_ylim(in_ylim)
    savefile = in_save_prefix + "_trace_" + in_var_name
    if not in_save_suffix is None:
        savefile += "_" + in_save_suffix
    savefile += ".png"
    fig_traces.savefig(savefile, bbox_inches="tight")
    plt.close(fig_traces)