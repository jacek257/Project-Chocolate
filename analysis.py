#!/usr/bin/env python


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.signal as sg
import scipy.interpolate as interp
from scipy.fftpack import fft, ifft
import os, sys

def getDataArray(f_path):
    df = pd.read_csv(f_path, sep='\t', names=['Time', 'O2', 'CO2', 'thrw', 'away'],
                 usecols=['Time', 'O2', 'CO2'], index_col=False)
    return df[["Time","O2","CO2"]]

def fourier_trans(t_step, data):
    """
    returns a tuple: (frequency domain, Power spectra, abs(power_spectra))

    t_step = temporal resolution of the time series (data)
    data = series to be analyzed
    """

    N = len(data)

    #create freq_dom from timestep
    freq_dom = np.linspace(0,1/(2*t_step),N//2)
    #perform fft
    power_spectra = fft(data)
    #abs(fft) cut in half
    plottable_spectra = (2/N * np.abs(power_spectra))[:N//2]
    return (freq_dom,power_spectra,plottable_spectra)

def filter(f_low, f_high, freq_dom, power_spectra):
    """
    returns power spectra that is cleaned of specific frequencies

    f_low = lower frequency bound
    f_high = upper frequency bound
    freq_dom = frequency domain as calculated by fourier_trans()
    power_spectra = power spectrum as calculated by fourier_trans()
    """
    #create a copy of the power_spectra
    cp = np.copy(power_spectra)

    #if f is between bounds, remove associated power
    for i,f in enumerate(freq_dom):
        if (f >= f_low and f<= f_high):
            cp[i] = 0
            cp[-i] = 0
    return np.copy(cp)

def fourier_filter(time_steps, data, low_f, high_f, TR):
    """
    Driver module: runs fourier_trans() and filter()

    time_steps = time_step list
    data = data to be analyzed
    low_f = lower frequency bound
    high_f = upper frequency bound
    TR = repetition time: found in BOLD .json
    """

    # fourier transform data
    freq, power, disp = fourier_trans(time_steps[1], data)
    #filter data
    pre_invert = filter(low_f,high_f, freq, power)
    #invert data and discard imaginary part
    inverted = ifft(pre_invert).real

    #convert time series to seconds
    if(time_steps[len(time_steps)-1]<10):
        time_steps = time_steps*60
    else:
        time_steps = time_steps

    #construct interpolation time_step series
    resample_ts = np.arange(0,480,TR)
    resampler = interp.interp1d(time_steps, inverted, fill_value="extrapolate")
    return (resampler(resample_ts))

def showMe(*plots):
    plt.figure(figsize=(20,10))
    for p in plots:
        plt.plot(p)
    plt.show()

def plotFourier(freq_dom, plottable):
    plt.figure(figsize=(20,10))
    plt.semilogx(x = freq_dom, y = plottable)

def main():
    f_path = sys.argv[1]
    processed_dir = os.path.dirname(f_path)+'/processed'
    print(processed_dir)
