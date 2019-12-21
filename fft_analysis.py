#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:09:10 2019

@author: Jimi
"""

import numpy as np
from scipy.fftpack import fft, ifft
import scipy.signal as sg
import scipy.interpolate as interp
import pandas as pd
import stat_utils

def fourier_trans(self, time_series, data, N):
    """
    Apply a fourier transformation on the data
    
    Parameters:
        time_series : pandas Series or array-like
            time series of the data that is be fourier transformed
        data : pandas Series or array-like
            actual data that will be fourier transformed
            
    Returns:
        tuple: (frequency domain, Power spectra, abs(power_spectra))
    """

    spacing = time_series.max()/N
    #create freq_dom from timestep
    freq_dom = np.linspace(0, 1/(2*spacing), N)
    #perform fft
    power_spectra = fft(data)
    #abs(fft) cut in half
    plottable_spectra = (2/N * np.abs(power_spectra))[:N//2]
    return (freq_dom,power_spectra,plottable_spectra)


def my_filter(self, f_cut, freq_dom, power_spectra):
    """
    Apply a butterworth low-pass filter on the fourier transformed data
    
    Parameters:
        f_low : float
            frequency cutoff
        freq_dom : array-like
            array of the frequencies
        power_spectra : array-like
            array of the power for each of the frequencies in the freq_dom
            
    Returns:
        array-like : power spectra that is cleaned of specific frequencies
    """
    #create a copy of the power_spectra
    cp = np.copy(power_spectra)

    # create filter
    b, a = sg.butter(11, f_cut, 'low', analog=True)
    w, h = sg.freqs(b, a)
    # extend filter
    resamp = interp.interp1d(w, h, fill_value='extrapolate')
    h = resamp(freq_dom)
    # apply filter
    for i,f in enumerate(freq_dom):
        cp[i] = cp[i] * np.abs(h)[i] if i < len(h) else 0
        
    return np.copy(cp)

def fourier_filter(self, time_series, data, f_cut, tr, time_points, trim):
    """
    Performs a fourier transform on the data and runs it through a buttersworth filter before
    applying an inverse fourier transform and interpolating to the new time.
    
    Parameters
        time_series : pandas Series or array-like
            The time series of the data
        data : pandas Series or array-like
            Data to be analyzed
        f_cut : float
            Frequency cut off
        tr : float
            Repetition time
        time_points : pandas Series or array-like
            Time for data to be interporlated onto

    Returns:
        pandas Dataframe that contains the Time and Data after filtering and interpolating
            
    """
    
    N = len(data)
    freq, power, disp = self.fourier_trans(time_series, data, N)
    
    # pass freq domain data through filter
    pre_invert = self.my_filter(f_cut, freq, power)
    inverted = ifft(pre_invert).real
    
    # create the target time series
    resample_ts = np.arange(time_series.min(), time_series.max()+tr, tr)
    # interpolate to match target time series
    resampler = interp.interp1d(time_series, inverted, fill_value="extrapolate")
    
    df = pd.DataFrame({'Time' : resample_ts,
                       'Data' : resampler(resample_ts)})

    if trim:
        df = stat_utils.trim_edges(df)
    
    return df


def fourier_filter_no_resample(self, time_series, data, f_cut, tr, trim):
    """
    Performs a fourier transform on the data and runs it through a buttersworth filter before
    applying an inverse fourier transform
    
    Parameters
        time_series : pandas Series or array-like
            The time series of the data
        data : pandas Series or array-like
            Data to be analyzed
        f_cut : float
            Frequency cut off
        tr : float
            Repetition time

    Returns:
        pandas Dataframe that contains the Time and Data after filtering and interpolating
            
    """
    
    N = len(data)
    freq, power, disp = self.fourier_trans(time_series, data, N)
    
    # pass freq domain data through filter
    pre_invert = self.my_filter(f_cut, freq, power)
    inverted = ifft(pre_invert).real
    
    df = pd.DataFrame({'Time' : time_series,
                       'Data' : inverted})

    if trim:
        df = stat_utils.trim_edges(df)
    
    return df
