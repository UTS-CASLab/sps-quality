# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:25:33 2023

@author: David J. Kedziora
"""

import numpy as np
from lmfit import Parameters, minimize

def func_pulsed(params, x, y, 
                use_poisson_likelihood = False, return_fit_error = False):
    # Define the pulsed fitting function for the delay histogram.
    # Is based on Eq. (2) of: https://doi.org/10.1063/1.5143786
    # x: domain (delays)
    # y: events per bin to be fitted
    # bg: amplitude correction for background number of events per bin (noise)
    # period_pulse: pulse period of the stimulating laser
    # delay_mpe: the delay value at which multi-photon emission (MPE) occurs
    # amp_env: amplitude of the envelope, i.e. the non-MPE peaks
    # amp_ratio: the ratio of the MPE peak to the envelope amplitude, i.e. g2(0)
    # factor_env: the inverted decay scale for the envelope amplitude 
    # decay_peak: the decay scale for an individual peak
    
    bg = params["bg"]
    period_pulse = params["period_pulse"]
    delay_mpe = params["delay_mpe"]
    amp_env = params["amp_env"]
    amp_ratio = params["amp_ratio"]
    factor_env = params["factor_env"]
    decay_peak = params["decay_peak"]
    
    tau = x - delay_mpe
    tau_m = np.mod(tau, period_pulse)
    
    fit = bg + amp_env*np.exp(-factor_env*np.abs(tau)) * (np.cosh((tau_m - period_pulse/2)/decay_peak) / np.sinh(period_pulse/(2*decay_peak)) 
       - (1 - amp_ratio)*np.exp(-np.abs(tau)/decay_peak))
    
    # See the supplementary material of: https://doi.org/10.1063/1.5143786
    # The error to minimise is based on Eq. (S25) and Eq. (S26).
    if return_fit_error:
        if use_poisson_likelihood:
            return np.sqrt(fit - y*np.log(fit))
        else:
            return fit - y
    else:
        return fit
    
def estimate_g2zero_pulsed(in_sr_sample, in_sr_delays, in_knowns, use_poisson_likelihood):
    
    period_pulse = in_knowns["period_pulse"]
    range_delays = in_sr_delays[in_sr_delays.size-1] - in_sr_delays[0]
    step_delays = in_sr_delays[1] - in_sr_delays[0]
    
    # If background dominates, the probability amplitude should be uniform.
    # If delta-function peaks dominate, amplitude will be their amount inverted. 
    amp_uniform = 1/len(in_sr_delays)
    amp_peaked = 1/np.floor(range_delays/period_pulse)
    
    params = Parameters()
    params.add("bg", value = amp_uniform, min = 0, max = 1)
    params.add("period_pulse", value = period_pulse, min = 0, max = np.inf, vary = False)
    params.add("delay_mpe", value = np.mean(in_knowns["delay_mpe"]), 
               min = in_knowns["delay_mpe"][0], max = in_knowns["delay_mpe"][-1])
    params.add("amp_env", value = amp_peaked, min = 0, max = 1)
    params.add("amp_ratio", value = 0.5, min = 0, max = 1)
    params.add("factor_env", value = 0, min = 0, max = np.inf, vary = False)
    params.add("decay_peak", value = step_delays, min = step_delays/10, max = np.inf)
    
    # Run a five-parameter fit with the Powell method.
    # Then run a five-parameter Levenberg-Marquardt fit, just for the errors.
    fitted_params = minimize(func_pulsed, params, 
                             args=(in_sr_delays, in_sr_sample, use_poisson_likelihood, True),
                             method="powell", calc_covar=False)
    # fitted_params = minimize(func_pulsed, fitted_params.params, 
    #                          args=(in_sr_delays, in_sr_sample, use_poisson_likelihood, True),
    #                          method="nelder_mead", calc_covar=False)

    # fitted_params.params["delay_mpe"].vary = False
    # fitted_params.params["bg"].vary = False
    # fitted_params.params["decay_peak"].vary = False
    # fitted_params = minimize(func_pulsed, fitted_params.params, 
    #                           args=(in_sr_delays, in_sr_sample, use_poisson_likelihood, True),
    #                           method="least_squares", calc_covar=False)
    
    # fitted_params.params["delay_mpe"].vary = True
    # fitted_params.params["bg"].vary = True
    # fitted_params.params["decay_peak"].vary = True
    fitted_params = minimize(func_pulsed, fitted_params.params, 
                              args=(in_sr_delays, in_sr_sample, use_poisson_likelihood, True),
                              method="least_squares")
    # fitted_params.params.pretty_print()
    
    return fitted_params