import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sg
import scipy.interpolate as interp
import seaborn as sns
import time
from tqdm import tqdm

#acknowledgements:
#https://ws680.nist.gov/publication/get_pdf.cfm?pub_id=901379

def get_centroid(t_series, data_series, window_coeff, poly_order):
    """
    centroid is a signal weighted average of time. We are essentially calculating the temporal middle of the signal

    t_series (iterable) = time points of data_series
    data_series (iterable) = the signal intensity series
    """
    #set buffer to 0
    buffer = 0.0
    #pack t_series and d_series together and calculate square weighted sum
    for t, d in zip(t_series, sg.savgol_filter(data_series, window_coeff, poly_order)):
        buffer += (d**2) * t
    #return square weighted average
    return buffer/(np.sum(data_series**2))

def align_centroid(base_time, base_sig, time1, sig1, time2, sig2): #TODO: arbitrary number of centroids
    """
    aligns several (3) time series based on their relative temporal centroids

    base_time = BOLD
    rest = self explanatory

    *time* (iterable) = time points / sampling points
    *sig* (iterable) = sampling intensity at each time/sampling points

    [*]sig[*] must have equal length to [*]time[*]

    returns a dictionary containing  {base_time, base_sig, sig1, sig2} properly aligned and interpolated
    """
    #get all centroids
    base_centroid = get_centroid(base_time, base_sig)
    sig1_centroid = get_centroid(time1, sig1)
    sig2_centroid = get_centroid(time2, sig2)

    #shift times and construct interpolators
    sig1_resampler = interp.interp1d(time1 - sig1_centroid + base_centroid, sig1)
    sig2_resampler = interp.interp1d(time2 - sig2_centroid + base_centroid, sig2)

    #interpolate signals
    sig1_exact_aligned = sig1_resampler(base_time)
    sig2_exact_aligned = sig2_resampler(base_time)

    return {'base_time' : base_time, 'base_sig' : base_sig, 'sig1' : sig1_exact_aligned, 'sig2' : sig2_exact_aligned}
