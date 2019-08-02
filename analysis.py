#!/usr/bin/env python


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.signal as sg
from scipy.fftpack import fft, ifft
import os, sys

def getDataArray(f_path):
    df = pd.read_csv(f_path, sep='\t', names=['Time', 'O2', 'CO2', 'thrw', 'away'],
                 usecols=['Time', 'O2', 'CO2'], index_col=False)
    return df[["Time","O2","CO2"]]

def fourier_trans(t_step, data):
    N = len(data)
    freq_dom = np.linspace(0,1/(2*t_step),N//2)
    power_spectra = fft(data)
    plottable_spectra = (2/N * np.abs(power_spectra))[:N//2]
    return (freq_dom,power_spectra,plottable_spectra)

def filter(f_low, f_high, freq_dom, power_spectra):
    cp = np.copy(power_spectra)
    for i,f in enumerate(freq_dom):
        if (f >= f_low and f<= f_high):
            cp[i] = 0
            cp[-i] = 0
    return np.copy(cp)

def fourier_filter(time_series, data, low_f, high_f):
    freq, power, disp = fourier_trans(time_series, data)
    pre_invert = filter(low_f,high_f, freq, power)
    return sg.resample(ifft(pre_invert).real,320)

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

main()

# processed_data_path = sys.arg[3]+'contrasts/'
# if(not os.path.exists(processed_data_path)):
#     os.mkdir(processed_data_path)
#     print ('path created')
#
# np.savetxt(processed_data_path+"contrast_O2.txt",downO.real, fmt = '%.18f', delimiter='\n')
# np.savetxt(processed_data_path+"contrast_CO2.txt", downC.real, fmt = '%.18f', delimiter='\n')
