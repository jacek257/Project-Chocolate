#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:25:25 2019

@author: Jimi Cao
"""
import numpy as np
import scipy.signal as sg
import pandas as pd
import stat_utils

def get_cross_correlation(base, sig, scan_time, ref_shift, invert):
    '''
    Parameters:
        base: pandas Series or array-like
            The base signal that the other signal will be shifted to matching
        sig : pandas Series or array-like
            Signal that is to be correlated
        scan_time : float
            Total scan time of the BOLD sequence
        ref_shift : int or None (default=None)
            The number of points from center that is set to be as the basis for the shift
            

    Returns:
        shift_val : float
            Time shift applied
        shift_index : int 
            Number of point shifts
        correlation_series : numpy array
            Cross_correlation values
    '''
#        plt.plot(base)
#        plt.plot(sig)
#        plt.show()
    correlation_series = sg.correlate(base, sig, mode='full')
    if ref_shift != None:
        limit = int(30 * len(base)/scan_time)
        lim_corr = correlation_series[len(correlation_series)//2-limit+ref_shift : len(correlation_series)//2+limit+1+ref_shift]
#            plt.plot(lim_corr)
#            plt.show()
        shift_index = np.argmax(lim_corr) - (len(lim_corr)//2) + ref_shift
    else:
        limit = int(10 * len(base)/scan_time)
        lim_corr = correlation_series[len(correlation_series)//2-limit : len(correlation_series)//2+limit+1]
        shift_index = np.argmax(lim_corr) - (len(lim_corr)//2)
#        shift_index = np.argmax(correlation_series) - (len(correlation_series)//2)
    shift_value = scan_time/len(base) * shift_index
        
    return shift_value, shift_index, correlation_series

def raw_align( base, raw_other, scan_time, time_points, ref_shift, invert):
    '''
    Parameters:
        base: pandas Series or array-like
            The base signal that the other signal will be shifted to match
        raw_other: pandas Dataframe or array-like
            Raw signal with the corresponding time to be correlated
        scan_time : float
            Total scan time of the BOLD sequence
        time_points : pandas Series or array-like
            Time series for the meants
        ref_shift : int or None (default=None)
            The number of points from center that is set to be as the basis for the shift
            

    Returns:
        df : pandas Dataframe 
            Shifted signal with corresponding time point
        shifted : numpy array 
            Shifted signal
        corr : numpy array 
            Cross_correlation values
        shift : float 
            Time shift applied
        start : int 
            Number of point shifts
    '''
    raw_padded = pad_zeros(raw_other.Data)
    shift, start, corr = get_cross_correlation(base, raw_padded, scan_time, ref_shift, invert)
    shifted = stat_utils.resamp(raw_other.Time+shift, raw_other.Time, raw_other.Data, shift, start)
        
    df = pd.DataFrame({ 'Time' : time_points,
                        'Data' : shifted})
    
    return df, shifted, corr, shift, start
    
def corr_align(base, other_time, other_sig, scan_time, time_points, ref_shift, invert):
    '''
    Parameters:
        base: pandas Series or array-like
            The base signal that the other signal will be shifted to matching
        other_time: pandas Series or array-like
            Time series of the signal to be correlated
        other_sig : pandas Series or array-like
            Signal that is to be correlated
        scan_time : float
            Total scan time of the BOLD sequence
        time_points : pandas Series or array-like
            Time series for the meants
        ref_shift : int or None (default=None)
            The number of points from center that is set to be as the basis for the shift
            

    Returns:
        df : pandas Dataframe of the shifted signal with corresponding time point
        shifted : numpy array of shifted signal
        corr : numpy array of cross_correlation values
        shift : float of the time shift applied
        start : int of the number of point shifts
    '''

#        other_padded = pad_zeros(other_sig)
    shift, start, corr = get_cross_correlation(base, other_sig, scan_time, ref_shift, invert)
    
    shifted = stat_utils.resamp(other_time+shift, time_points, other_sig, shift, start)
    
    df = pd.DataFrame({ 'Time' : time_points,
                        'Data' : shifted})
    
    return df, shifted, corr, shift, start
    
def pad_zeros(sig):
    pre = np.zeros_like(sig)
    post = np.zeros_like(sig)
    padded = np.concatenate((pre, sig, post))
    return padded
