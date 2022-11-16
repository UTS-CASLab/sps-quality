# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:52:15 2022

@author: David J. Kedziora
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_traces(in_trace_matrix, in_axis_time, 
                in_save_prefix, in_var_name, in_save_suffix = None, 
                in_ylim = None):
    
    # Input trace matrix should have a row for each trajectory.
    # This code transposes it.
    fig_traces, ax_traces = plt.subplots()
    ax_traces.plot(in_axis_time, np.transpose(in_trace_matrix))
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