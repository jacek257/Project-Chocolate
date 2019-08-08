#!/usr/bin/env python


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    
def get_peaks(df, verb, file, TR):
    """
    Get the peaks and troughs of CO2 and O2
    
    Parameters:
        df: dataframe
            data structure that holds the data
        verb: boolean
            flag for verbose output
        file: string
            file name to be displayed during verbose output
        TR: float
            repetition time, found in BOLD.json
    
    Returns:
        et_O2: array-like
            the end-tidal O2 data
        et_CO2: array-like
            the end-tidal CO2 data
    """    
    
    # set the size of the graphs
    sns.set(rc={'figure.figsize':(20,10)})
    
    #convert time series to seconds
    if(df.Time[len(df)-1]<10):
        df.Time = df.Time*60
    
    #construct interpolation time_step series
    resample_ts = np.arange(0,480,TR)
    
    # make another loop for user confirmation that CO2 peak detection is good
    bad = True
    prom = 1

    while bad:
        # get the troughs of the O2 data
        low_O2, _ = sg.find_peaks(df.O2.apply(lambda x:x*-1), prominence=prom)

        # create scatterplot of all O2 data
        if verb:
            print("Creating O2 plot ", file)
        sns.lineplot(x='Time', y='O2', data=df, linewidth=1, color='b')

        # get the data points of peak
        O2_df = df.iloc[low_O2]
        
        # linear interpolate the number of data points to match resample_ts
        O2_fxn = interp.interp1d(O2_df.Time, O2_df.O2, fill_value='extrapolate')
        et_O2 = O2_fxn(resample_ts)

        # add peak overlay onto the scatterplot
        sns.lineplot(x=resample_ts, y=et_O2, linewidth=2, color='g')
        plt.show()
        plt.close()

        # ask user if the peak finding was good
        ans = input("Was the output good enough (y/n)? \nNote: anything not starting with 'y' is considered 'n'.\n")
        bad = True if ans == '' or ans[0].lower() != 'y' else False
        if bad:
            print("The following variables can be changed: ")
            print("    1. prominence - Required prominence of peaks. Type: int")
            try:
                prom = int(input("New prominence (Default is 1): "))
            except:
                print("Default value used")
                prom = 1


    # make another loop for user confirmation that CO2 peak detection is good
    bad = True
    prom = 1
    while bad:
        # get peaks of the CO2 data
        high_CO2, _ = sg.find_peaks(df.CO2, prominence=prom)

        # create scatter of all CO2 data
        if verb:
            print('Creating CO2 plot ', file)
        sns.lineplot(x='Time', y='CO2', data=df, linewidth=1, color='b')

        # get the data points of peak
        CO2_df = df.iloc[high_CO2]

        # linear interpolate the number of data points to match the scan Time
        CO2_fxn = interp.interp1d(CO2_df.Time, CO2_df.CO2, fill_value='extrapolate')
        et_CO2 = CO2_fxn(resample_ts)

        # add peak overlay onto the scatterplot
        sns.lineplot(x=resample_ts, y=et_CO2, linewidth=2, color='r')
        plt.show()
        plt.close()

        # ask user if the peak finding was good
        ans = input("Was the output good enough (y/n)? \nNote: anything not starting with 'y' is considered 'n'.\n")
        bad = True if ans == '' or ans[0].lower() != 'y' else False
        if bad:
            print("The following variables can be changed: ")
            print("    1. prominence - Required prominence of peaks. Type: int")
            try:
                prom = int(input("New prominence (Default is 1): "))
            except:
                print("Default value used")
                prom = 1

    plt.close()
    
    return et_CO2, et_O2

def save_plots(df, O2, CO2, f_path, verb, TR):
    """
    Create and saves plots for CO2 and O2 data
    
    Parameters:
        df: dataframe
            data structure that holds the data
        verb: boolean
            flag for verbose output
        f_path: string
            the path of the file
        verb: boolean
            flag for verbose output
        TR: float
            repitition time from json
    
    Returns:
        None
    """
    
    # set the size of the graphs
    sns.set(rc={'figure.figsize':(20,10)})
    
    #construct interpolation time_step series
    resample_ts = np.arange(0,480,TR)

    # create subplots for png file later
    f, axes = plt.subplots(2, 1)

    # recreate the plot because plt.show clears plt
    sns.lineplot(x='Time', y='O2', data=df, linewidth=1, color='b', ax=axes[0])
    sns.lineplot(x=resample_ts, y=O2, linewidth=2, color='g', ax=axes[0])
    # recreate the plot because plt.show clears plt
    sns.lineplot(x='Time', y='CO2', data=df, linewidth=1, color='b', ax=axes[1])
    sns.lineplot(x=resample_ts, y=CO2, linewidth=2, color='r', ax=axes[1])

    # save the plot
    if verb:
        print('Saving plots for', f_path)
        
    save_path = save_path = f_path[:-4]+'/graph.png'
    f.savefig(save_path)
    if verb:
        print('Saving complete')
    f.clf()

    if verb:
        print()

def main():
    f_path = sys.argv[1]
    processed_dir = os.path.dirname(f_path)+'/processed'
    print(processed_dir)
