#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:19:52 2019

@author: Jimi Cao
"""
import numpy as np
import scipy.signal as sg
import scipy.interpolate as interp
import pandas as pd
import stat_utils
import fft_analysis
import matplotlib.pyplot as plt
import seaborn as sns

def peak_four(df, verb, file, tr, time_pts, trough):
    
    f_CO2 = fft_analysis.fourier_filter_no_resample(df.Time, df.CO2, 1/60, tr, trim=True)
    
    resamp_tp = np.arange(0, df.Time.max(), tr)

    # get the troughs of the O2 data
    O2_data, _ = sg.find_peaks(df.O2.apply(lambda x:x*-1), prominence=2)
    O2_df = df.drop(columns=['CO2'])
    O2_df = O2_df.iloc[O2_data]
    
    O2_valid_df = O2_df
    
    
    O2_resamp = interp.interp1d(O2_valid_df.Time, O2_valid_df.O2, fill_value='extrapolate')
    O2_final_df = pd.DataFrame({'Time' : resamp_tp,
                                'Data' : O2_resamp(resamp_tp)})
    
    O2_final_df = stat_utils().trim_edges(O2_final_df)
    
    if trough:
        CO2_data, _ = sg.find_peaks(df.CO2.apply(lambda x:x*-1), prominence=3, width=20)
    else:
        CO2_data, _ = sg.find_peaks(df.CO2, prominence=3, width=30) 
    
    CO2_df = df.drop(columns=['O2'])
    CO2_df = CO2_df.iloc[CO2_data]
    
    f_CO2_resamp = interp.interp1d(f_CO2.Time, f_CO2.Data , fill_value='extrapolate')
    f_CO2_final = f_CO2_resamp(CO2_df.Time)
    
    
    if trough:
        CO2_df['cmp'] = CO2_df.CO2 < f_CO2_final
    else:
        CO2_df['cmp'] = CO2_df.CO2 > f_CO2_final
    
    CO2_valid_df = CO2_df[CO2_df.cmp == True].reset_index(drop=True)
    
    CO2_resamp = interp.interp1d(CO2_valid_df.Time, CO2_valid_df.CO2, fill_value='extrapolate')
    CO2_final_df = pd.DataFrame({'Time' : resamp_tp,
                                 'Data' : CO2_resamp(resamp_tp)})
    
    CO2_final_df = stat_utils().trim_edges(CO2_final_df)
    return CO2_final_df, O2_final_df


def peak(df, verb, file, time_points):
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
    
    # make loop for user confirmation that O2 peak detection is good
    bad = True
    prom = 2

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
        O2_final_df = pd.DataFrame({'Time' : df.Time,
                                    'Data' : O2_fxn(df.Time)})

        # add peak overlay onto the scatterplot
        sns.lineplot(x=O2_final_df.Time, y=O2_final_df.Data, linewidth=2, color='g')
        plt.show()
        plt.close()

        # ask user if the peak finding was good
        ans = input("Was the output good enough (y/n)? \nNote: anything not starting with 'y' is considered 'n'.\n")
        bad = True if ans == '' or ans[0].lower() != 'y' else False
        if bad:
            print("The following variables can be changed: ")
            print("    1. prominence - Required prominence of peaks. Type: int")
            try:
                prom = int(input("New prominence (Default is 2): "))
            except:
                print("Default value used")
                prom = 2


    # make another loop for user confirmation that CO2 peak detection is good
    bad = True
    prom = 3
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
        CO2_final_df = pd.DataFrame({'Time' : df.Time,
                                     'Data' : CO2_fxn(df.Time)})

        # add peak overlay onto the scatterplot
        sns.lineplot(x=CO2_final_df.Time, y=CO2_final_df.Data, linewidth=2, color='r')
        plt.show()
        plt.close()

        # ask user if the peak finding was good
        ans = input("Was the output good enough (y/n)? \nNote: anything not starting with 'y' is considered 'n'.\n")
        bad = True if ans == '' or ans[0].lower() != 'y' else False
        if bad:
            print("The following variables can be changed: ")
            print("    1. prominence - Required prominence of peaks. Type: int")
            try:
                prom = int(input("New prominence (Default is 3): "))
            except:
                print("Default value used")
                prom = 3
                
    plt.close()
    
    return CO2_final_df, O2_final_df

def get_wlen(sig_time, sig):

    freq,_,disp = fft_analysis.fourier_trans(sig_time, sig, len(sig))
    window_it = np.argmax(disp[25:500])+25
#    plt.close()
#    sns.lineplot(data=disp[:500])
#    plt.show()
#    plt.close()
    freq_val = freq[window_it] # this is the most prominent frequency
#    print(freq_val)
    window_mag = 1/freq_val
#    print(window_mag)
    if window_mag > 10:
        window_mag /= 2
#    print(window_mag)

    window_length = 0
    for i, t in enumerate(sig_time):
        if t > window_mag:
            window_length = i-1
            break

    return window_length


def envelope(sig_time, sig, tr, invert):
    """
    Params:
        sigtime (iterable) = the sampling time points of sig
        sig (iterable) = signal to perform peakfinding and resampling on

    Return: (iterable)
        Peak found time-series which aligns with base_timeit
    """
    time_pts = np.arange(sig_time.min(), sig_time.max()+tr, tr)
    cpy = pd.Series(sig)
    if invert:
        cpy *= -1
    cpy = cpy.reset_index(drop=True)
    count = 0

    window_length = get_wlen(sig_time, cpy)
    for i in range(0,len(cpy),window_length):
        for j in range(i, i+window_length):
            if j < len(cpy):
                cpy[j] = cpy[cpy[i:i+window_length].idxmax()]
        count += 1
    if invert:
        cpy *= -1
#    plt.close()
#    sns.lineplot(x=sig_time, y=sig)
#    sns.lineplot(x=sig_time, y=cpy, color='r')
#    plt.show()
#    plt.close()
    
    # get the sampling freq
    fs = len(cpy)/np.max(sig_time)
    # get cutoff freq
    fc = count / np.max(sig_time)
    w = fc / (fs / 2)

    b, a = sg.butter(1, w, 'low', analog=False)
    filtered = sg.filtfilt(b, a, cpy)
    signal_resampled = stat_utils.resamp(sig_time, time_pts, filtered, 0, 0)

#    sns.lineplot(x=sig_time, y=sig)
#    sns.lineplot(x=time_pts, y=signal_resampled, color='r')
#    plt.show()
#    plt.close()
    return pd.DataFrame({'Time' : time_pts,
                         'Data' : signal_resampled})
