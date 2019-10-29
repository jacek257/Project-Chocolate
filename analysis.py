#!/usr/bin/env python


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.signal as sg
import scipy.interpolate as interp
from scipy.fftpack import fft, ifft
from tqdm import tqdm
import sys
import time
import subprocess
import os
import signal
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.linear_model import Ridge
from scipy import stats

class fft_analysis:
    """docstring for fft_analysis."""

    def fourier_trans(self, time_series, data, N):
        """
        returns a tuple: (frequency domain, Power spectra, abs(power_spectra))

        spacing = the distance between data points
        data = series to be analyzed
        """

        spacing = time_series.max()/N
        #create freq_dom from timestep
        freq_dom = np.linspace(0, 1/(2*spacing), N)
        #perform fft
        power_spectra = fft(data)
        #abs(fft) cut in half
        plottable_spectra = (2/N * np.abs(power_spectra))[:N//2]
        return (freq_dom,power_spectra,plottable_spectra)


    def my_filter(self, f_low, f_high, freq_dom, power_spectra):
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
#        for i,f in enumerate(freq_dom):
##            if (f >= f_low) and (f <= f_high):
##                cp[i] = 0
##                cp[-i] = 0
#            if f >= f_low:
#                cp[i] = 0

        b, a = sg.butter(11, f_low, 'low', analog=True)
        w, h = sg.freqs(b, a)
        resamp = interp.interp1d(w, h, fill_value='extrapolate')
        h = resamp(freq_dom)
        for i,f in enumerate(freq_dom):
            cp[i] = cp[i] * np.abs(h)[i] if i < len(h) else 0
            
        return np.copy(cp)

    def fourier_filter(self, time_series, data, low_f, high_f, tr, time_points, trim):
        """
        Driver module: runs fourier_trans() and my_filter() and downsamples

        time_steps = time_step list
        data = data to be analyzed
        low_f = lower frequency bound
        high_f = upper frequency bound
        TR = repetition time: found in BOLD .json
        """
        
        N = len(data)
        freq, power, disp = self.fourier_trans(time_series, data, N)
        
        pre_invert = self.my_filter(low_f,high_f, freq, power)
        inverted = ifft(pre_invert).real
        
        resample_ts = np.arange(time_series.min(), time_series.max()+tr, tr)
#        print(resample_ts)
        resampler = interp.interp1d(time_series, inverted, fill_value="extrapolate")
        
#        df = pd.DataFrame({'Time' : time_series,
#                           'Data' : inverted})
    
        df = pd.DataFrame({'Time' : resample_ts,
                           'Data' : resampler(resample_ts)})
    
        if trim:
            df = stat_utils().trim_edges(df)
        
#        df = df[df.Time > 5].reset_index(drop=True)
#        df = df[df.Time < df.Time.max()-5].reset_index(drop=True)
        
        return df


    def fourier_filter_no_resample(self, time_series, data, low_f, high_f):
        """
        Driver module: runs fourier_trans() and my_filter() and does not downsample

        time_steps = time_step list
        data = data to be analyzed
        low_f = lower frequency bound
        high_f = upper frequency bound
        """
        freq, power, disp = self.fourier_trans(time_series[1], data)
        pre_invert = self.my_filter(low_f,high_f, freq, power)
        inverted = ifft(pre_invert).real
        return pd.DataFrame({'Time' : time_series,
                             'Data' : inverted})

class stat_utils:
    """docstring for stat_utils."""


    def save_plots_comb_only(self, df, O2, O2_f, 
                   CO2, CO2_f, meants,
                   coeff, coeff_f, comb_corr,
                   f_path, key, verb, time_points, TR):
        
        if verb:
            print('Creating regression plot')
        
        # set the size of the graphs
        sns.set(rc={'figure.figsize':(30,20)})
        plt.rc('legend', fontsize='x-large')
        plt.rc('xtick', labelsize='x-large')
        plt.rc('axes', titlesize='x-large')

        # normalize data
        meants_norm = sg.detrend(meants)
        meants_norm /= meants_norm.std()
#        sns.lineplot(x=time_points, y=meants_norm, color='blue', ax=axes[0])
#        predict = coeff[0]*O2_shift + coeff[1]*CO2_shift + coeff[2]
#        predict_norm = sg.detrend(predict)
#        predict_norm /= predict_norm.std()
#        sns.lineplot(x=time_points, y=predict_norm, color='violet', ax=axes[0])
        
        f, axes = plt.subplots(3, 1)
        
        sns.lineplot(x=time_points, y=meants_norm, color='blue', ax=axes[0])
        predict = coeff[0]*O2 + coeff[1]*CO2 + coeff[2]
        predict_norm = sg.detrend(predict)
        predict_norm /= predict_norm.std()
        sns.lineplot(x=time_points, y=predict_norm[:len(meants)], color='black', ax=axes[0])
        axes[0].set_title('No-shift vs BOLD')
        axes[0].legend(['BOLD', 'No-shift'], facecolor='w')
        
        sns.lineplot(x=np.arange(-len(comb_corr)//2, len(comb_corr)//2)+1, y=comb_corr, ax=axes[1])
        axes[1].set_title('Cross-Correlation')
        axes[1].legend(['Cross-Correlation'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='blue', ax=axes[2])
        predict = coeff_f[0]*O2_f + coeff_f[1]*CO2_f + coeff_f[2]
        predict_norm = sg.detrend(predict)
        predict_norm /= predict_norm.std()
        sns.lineplot(x=time_points, y=predict_norm, color='black', ax=axes[2])
        axes[2].set_title('With-shift vs BOLD')
        axes[2].legend(['BOLD', 'With-shift'], facecolor='w')

        if verb:
            print('Saving regression plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'regression.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        return
    
    def save_plots_edge(self, df, O2_time, O2, O2_shift, O2_shift_f,
                        CO2_time, CO2, CO2_shift, CO2_shift_f, meants,
                        coeff, coeff_f,
                        f_path, key, verb, time_points, TR):

        # set the size of the graphs
        sns.set(rc={'figure.figsize':(30,20)})
        plt.rc('legend', fontsize='x-large')
        plt.rc('xtick', labelsize='x-large')
        plt.rc('axes', titlesize='x-large')

        # normalize data
        meants_norm = sg.detrend(meants)
        meants_norm /= meants_norm.std()

        CO2_norm = sg.detrend(CO2)
        CO2_norm /= CO2_norm.std()

        CO2_shift_norm = sg.detrend(CO2_shift)
        CO2_shift_norm /= CO2_shift_norm.std()

        O2_norm = sg.detrend(O2)
        O2_norm /= O2_norm.std()

        O2_shift_norm = sg.detrend(O2_shift)
        O2_shift_norm /= O2_shift_norm.std()

        if verb:
            print('Creating O2 plots to be saved')
        # create subplots for png file later
        f, axes = plt.subplots(3, 1)

        sns.lineplot(x='Time', y='O2', data=df, linewidth=1, color='b', ax=axes[0])
        sns.lineplot(x=O2_time, y=O2, linewidth=2, color='g', ax=axes[0])
        axes[0].set_title('Processed vs Raw')
        axes[0].legend(['Raw', 'Processed'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1])
        sns.lineplot(x=O2_time, y=O2_norm, color='g', ax=axes[1])
        axes[1].set_title('Processed vs BOLD')
        axes[1].legend(['BOLD', 'Processed'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[2])
        sns.lineplot(x=time_points, y=O2_shift_norm, color='g', ax=axes[2])
        axes[2].set_title('Shifted vs BOLD')
        axes[2].legend(['BOLD', 'Shifted'], facecolor='w')

        # save the plot
        if verb:
            print('Saving O2 plots for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'O2_graph.png'
        f.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)

        if verb:
            print('Creating CO2 plots to be saved')
        # create subplots for png file later
        f, axes = plt.subplots(3, 1)

        sns.lineplot(x='Time', y='CO2', data=df, linewidth=1, color='b', ax=axes[0])
        sns.lineplot(x=O2_time, y=CO2, linewidth=2, color='r', ax=axes[0])
        axes[0].set_title('Processed vs Raw')
        axes[0].legend(['Raw', 'Processed'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1])
        sns.lineplot(x=CO2_time, y=CO2_norm, color='r', ax=axes[1])
        axes[1].set_title('Processed vs BOLD')
        axes[1].legend(['BOLD', 'Processed'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[2])
        sns.lineplot(x=time_points, y=CO2_shift_norm, color='r', ax=axes[2])
        axes[2].set_title('Shifted vs BOLD')
        axes[2].legend(['BOLD', 'Shifted'], facecolor='w')

        # save the plot
        if verb:
            print('Saving CO2 plots for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'CO2_graph.png'
        f.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        if verb:
            print('Creating regression plot')     
        sns.set(rc={'figure.figsize':(30,20)})   
        f, axes = plt.subplots(2, 1)
        
        sns.lineplot(x=time_points, y=meants_norm, color='blue', ax=axes[0])
        predict = coeff[0]*O2_shift + coeff[1]*CO2_shift + coeff[2]
        predict_norm = sg.detrend(predict)
        predict_norm /= predict_norm.std()
        sns.lineplot(x=time_points, y=predict_norm, color='violet', ax=axes[0])
        axes[0].set_title('First-shift vs BOLD')
        axes[0].legend(['BOLD', 'First-shift'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='blue', ax=axes[1])
        predict = coeff_f[0]*O2_shift_f + coeff_f[1]*CO2_shift_f + coeff_f[1]
        predict_norm = sg.detrend(predict)
        predict_norm /= predict_norm.std()
        sns.lineplot(x=time_points, y=predict_norm, color='violet', ax=axes[1])
        axes[1].set_title('Second-shift vs BOLD')
        axes[1].legend(['BOLD', 'Second-shift'], facecolor='w')

        if verb:
            print('Saving regression plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'regression.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        return       
    
    def save_plots_no_comb(self, df, O2, O2_f, O2_corr,
                           CO2, CO2_f, CO2_corr, meants,
                           f_path, key, verb, time_points):
        
        if verb:
            print('Creating O2 plots')
        
        # set the size of the graphs
        sns.set(rc={'figure.figsize':(30,20)})
        plt.rc('legend', fontsize='medium')
        plt.rc('xtick', labelsize='medium')
        plt.rc('ytick', labelsize='x-small')
        plt.rc('axes', titlesize='medium')

        meants_norm = meants / meants.std()
        O2_norm = O2.Data / O2.Data.std()
        O2_f_norm = O2_f.Data / O2.Data.std()
        
        f, axes = plt.subplots(2, 2)
        
        sns.lineplot(x='Time', y='O2', data=df, color='b', ax=axes[0, 0])
        sns.lineplot(x='Time', y='Data', data=O2, color='g', ax=axes[0, 0])
        axes[0, 0].set_title('Raw O2 vs Processed O2')
        axes[0, 0].legend(['Raw O2', 'Processed O2'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0, 1])
        sns.lineplot(x='Time', y=O2_norm, data=O2, color='g', ax=axes[0, 1])
        axes[0, 1].set_title('Processed O2 vs BOLD')
        axes[0, 1].legend(['BOLD', 'Processed O2'], facecolor='w')
        
        sns.lineplot(x=np.arange(-len(O2_corr)//2, len(O2_corr)//2)+1, y=O2_corr, ax=axes[1, 0])
        axes[1, 0].set_title('Cross-Correlation')
        axes[1, 0].legend(['Cross-Correlation'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1, 1])
        sns.lineplot(x='Time', y=O2_f_norm, data=O2_f, color='g', ax=axes[1, 1])
        axes[1, 1].set_title('Shifted O2 vs BOLD')
        axes[1, 1].legend(['BOLD', 'Shifted O2'], facecolor='w')

        if verb:
            print('Saving O2 plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'O2.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        if verb:
            print('Creating CO2 plots')
            
        f, axes = plt.subplots(2, 2)
        
        CO2_norm = CO2.Data / CO2.Data.std()
        CO2_f_norm = CO2_f.Data / CO2.Data.std()
        
        sns.lineplot(x='Time', y='CO2', data=df, color='b', ax=axes[0, 0])
        sns.lineplot(x='Time', y='Data', data=CO2, color='r', ax=axes[0, 0])
        axes[0, 0].set_title('Raw CO2 vs Processed CO2')
        axes[0, 0].legend(['Raw O2', 'Processed O2'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0, 1])
        sns.lineplot(x='Time', y=CO2_norm, data=CO2, color='r', ax=axes[0, 1])
        axes[0, 1].set_title('Processed CO2 vs BOLD')
        axes[0, 1].legend(['BOLD', 'Processed CO2'], facecolor='w')
        
        sns.lineplot(x=np.arange(-len(CO2_corr)//2, len(CO2_corr)//2)+1, y=CO2_corr, ax=axes[1, 0])
        axes[1, 0].set_title('Cross-Correlation')
        axes[1, 0].legend(['Cross-Correlation'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1, 1])
        sns.lineplot(x='Time', y=CO2_f_norm, data=CO2_f, color='r', ax=axes[1, 1])
        axes[1, 1].set_title('BOLD vs Shifted CO2')
        axes[1, 1].legend(['BOLD', 'Shifted CO2'], facecolor='w')

        if verb:
            print('Saving CO2 plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'CO2.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        if verb:
            print('Creating stacked plot')
        
        f, axes = plt.subplots(2, 1)
        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0])
        sns.lineplot(x='Time', y=O2_norm, data=O2, color='g', ax=axes[0])
        sns.lineplot(x='Time', y=CO2_norm, data=CO2, color='r', ax=axes[0])
        axes[0].set_title('Processed Gases vs BOLD')
        axes[0].legend(['BOLD','Processed O2', 'Processed CO2'], facecolor='w')
        
        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1])
        sns.lineplot(x='Time', y=O2_f_norm, data=O2, color='g', ax=axes[1])
        sns.lineplot(x='Time', y=CO2_f_norm, data=CO2, color='r', ax=axes[1])
        axes[1].set_title('Shifted Gases vs BOLD')
        axes[1].legend(['BOLD','Shifted O2', 'Shifted CO2'], facecolor='w')

        if verb:
            print('Saving stacked plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'stacked.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        return    
    
    def save_plots_comb(self, df, O2, O2_m, O2_f, O2_corr,
                        CO2, CO2_m, CO2_f, CO2_corr, meants,
                        coeff, coeff_f, comb_corr,
                        f_path, key, verb, time_points):
        
        if verb:
            print('Creating O2 plots')
        
        # set the size of the graphs
        sns.set(rc={'figure.figsize':(30,20)})
        plt.rc('legend', fontsize='medium')
        plt.rc('xtick', labelsize='medium')
        plt.rc('ytick', labelsize='x-small')
        plt.rc('axes', titlesize='medium')

        meants_norm = meants / meants.std()
        O2_norm = O2.Data / O2.Data.std()
        O2_m_norm = O2_m.Data / O2_m.Data.std()
        
        f, axes = plt.subplots(2, 2)
        
        sns.lineplot(x='Time', y='O2', data=df, color='b', ax=axes[0, 0])
        sns.lineplot(x='Time', y='Data', data=O2, color='g', ax=axes[0, 0])
        axes[0, 0].set_title('Raw O2 vs Processed O2')
        axes[0, 0].legend(['Raw O2', 'Processed O2'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0, 1])
        sns.lineplot(x='Time', y=O2_norm, data=O2, color='g', ax=axes[0, 1])
        axes[0, 1].set_title('Processed O2 vs BOLD')
        axes[0, 1].legend(['BOLD', 'Processed O2'], facecolor='w')
        
        sns.lineplot(x=np.arange(-len(O2_corr)//2, len(O2_corr)//2)+1, y=O2_corr, ax=axes[1, 0])
        axes[1, 0].set_title('Cross-Correlation')
        axes[1, 0].legend(['Cross-Correlation'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1, 1])
        sns.lineplot(x='Time', y=O2_m_norm, data=O2_m, color='g', ax=axes[1, 1])
        axes[1, 1].set_title('Shifted O2 vs BOLD')
        axes[1, 1].legend(['BOLD', 'Shifted O2'], facecolor='w')

        if verb:
            print('Saving O2 plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'O2.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        if verb:
            print('Creating CO2 plots')
            
        f, axes = plt.subplots(2, 2)
        
        CO2_norm = CO2.Data / CO2.Data.std()
        CO2_m_norm = CO2_m.Data / CO2_m.Data.std()
        
        sns.lineplot(x='Time', y='CO2', data=df, color='b', ax=axes[0, 0])
        sns.lineplot(x='Time', y='Data', data=CO2, color='r', ax=axes[0, 0])
        axes[0, 0].set_title('Raw CO2 vs Processed CO2')
        axes[0, 0].legend(['Raw O2', 'Processed O2'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0, 1])
        sns.lineplot(x='Time', y=CO2_norm, data=CO2, color='r', ax=axes[0, 1])
        axes[0, 1].set_title('Processed CO2 vs BOLD')
        axes[0, 1].legend(['BOLD', 'Processed CO2'], facecolor='w')
        
        sns.lineplot(x=np.arange(-len(CO2_corr)//2, len(CO2_corr)//2)+1, y=CO2_corr, ax=axes[1, 0])
        axes[1, 0].set_title('Cross-Correlation')
        axes[1, 0].legend(['Cross-Correlation'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1, 1])
        sns.lineplot(x='Time', y=CO2_m_norm, data=CO2_m, color='r', ax=axes[1, 1])
        axes[1, 1].set_title('BOLD vs Shifted CO2')
        axes[1, 1].legend(['BOLD', 'Shifted CO2'], facecolor='w')

        if verb:
            print('Saving CO2 plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'CO2.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        if verb:
            print('Creating combined plots')
            
        f, axes = plt.subplots(3, 1)
        
        O2_f_norm = O2_f / O2_f.std()
        CO2_f_norm = CO2_f / CO2_f.std()
        combined = coeff[0] * O2_m_norm + coeff[1] * CO2_m_norm + coeff[2]
        
        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0])
        sns.lineplot(x='Time', y=combined, data=CO2, color='black', ax=axes[0])
        axes[0].set_title('BOLD vs Combination')
        axes[0].legend(['BOLD', 'Combination'], facecolor='w')
        
        sns.lineplot(x=np.arange(-len(comb_corr)//2, len(comb_corr)//2)+1, y=comb_corr, ax=axes[1])
        axes[1].set_title('Cross-Correlation')
        axes[1].legend(['Cross-Correlation'], facecolor='w')

        combined_s = coeff_f[0] * O2_f_norm + coeff_f[1] * CO2_f_norm + coeff_f[2]
        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[2])
        sns.lineplot(x=time_points, y=combined_s, color='black', ax=axes[2])
        axes[2].set_title('BOLD vs Shifted Combination')
        axes[2].legend(['BOLD', 'Shifted Combination'], facecolor='w')

        if verb:
            print('Saving combined plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'combined.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        if verb:
            print('Creating stacked plot')
        
        f, axes = plt.subplots(2, 1)
        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0])
        sns.lineplot(x='Time', y=O2_m_norm, data=O2_m, color='g', ax=axes[0])
        sns.lineplot(x='Time', y=CO2_m_norm, data=CO2_m, color='r', ax=axes[0])
        sns.lineplot(x=time_points, y=combined, color='black', ax=axes[0])
        axes[0].set_title('Processed Gases vs BOLD')
        axes[0].legend(['BOLD','Processed O2', 'Processed CO2', 'Combination'], facecolor='w')
        
        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1])
        sns.lineplot(x=time_points, y=O2_f_norm, color='g', ax=axes[1])
        sns.lineplot(x=time_points, y=CO2_f_norm, color='r', ax=axes[1])
        sns.lineplot(x=time_points, y=combined_s, color='black', ax=axes[1])
        axes[1].set_title('Shifted Gases vs BOLD')
        axes[1].legend(['BOLD','Shifted O2', 'Shifted CO2', 'Shifted Combination'], facecolor='w')

        if verb:
            print('Saving stacked plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'stacked.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        return    
    
    def save_plots_no_comb_raw(self, df, O2, pre_O2, O2_f, O2_corr,
                           CO2, CO2_f, pre_CO2, CO2_corr, meants,
                           f_path, key, verb, time_points):
        
        if verb:
            print('Creating O2 plots')
        
        # set the size of the graphs
        sns.set(rc={'figure.figsize':(30,20)})
        plt.rc('legend', fontsize='medium')
        plt.rc('xtick', labelsize='medium')
        plt.rc('ytick', labelsize='x-small')
        plt.rc('axes', titlesize='medium')

        meants_norm = meants / meants.std()
        O2_norm = O2.Data / O2.Data.std()
        O2_f_norm = O2_f.Data / O2.Data.std()
        
        f, axes = plt.subplots(2, 2)
        
        sns.lineplot(x='Time', y='O2', data=df, color='b', ax=axes[0, 0])
        sns.lineplot(x='Time', y='Data', data=pre_O2, color='g', ax=axes[0, 0])
        axes[0, 0].set_title('Raw O2 vs Processed O2')
        axes[0, 0].legend(['Raw O2', 'Processed O2'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0, 1])
        sns.lineplot(x='Time', y=O2_norm, data=O2, color='g', ax=axes[0, 1])
        axes[0, 1].set_title('Raw O2 vs BOLD')
        axes[0, 1].legend(['BOLD', 'Raw O2'], facecolor='w')
        
        sns.lineplot(x=np.arange(-len(O2_corr)//2, len(O2_corr)//2)+1, y=O2_corr, ax=axes[1, 0])
        axes[1, 0].set_title('Cross-Correlation')
        axes[1, 0].legend(['Cross-Correlation'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1, 1])
        sns.lineplot(x='Time', y=O2_f_norm, data=O2_f, color='g', ax=axes[1, 1])
        axes[1, 1].set_title('Shifted O2 vs BOLD')
        axes[1, 1].legend(['BOLD', 'Shifted O2'], facecolor='w')

        if verb:
            print('Saving O2 plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'O2.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        if verb:
            print('Creating CO2 plots')
            
        f, axes = plt.subplots(2, 2)
        
        CO2_norm = CO2.Data / CO2.Data.std()
        CO2_f_norm = CO2_f.Data / CO2.Data.std()
        
        sns.lineplot(x='Time', y='CO2', data=df, color='b', ax=axes[0, 0])
        sns.lineplot(x='Time', y='Data', data=pre_CO2, color='r', ax=axes[0, 0])
        axes[0, 0].set_title('Raw CO2 vs Processed CO2')
        axes[0, 0].legend(['Raw O2', 'Processed O2'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0, 1])
        sns.lineplot(x='Time', y=CO2_norm, data=CO2, color='r', ax=axes[0, 1])
        axes[0, 1].set_title('Raw CO2 vs BOLD')
        axes[0, 1].legend(['BOLD', 'Raw CO2'], facecolor='w')
        
        sns.lineplot(x=np.arange(-len(CO2_corr)//2, len(CO2_corr)//2)+1, y=CO2_corr, ax=axes[1, 0])
        axes[1, 0].set_title('Cross-Correlation')
        axes[1, 0].legend(['Cross-Correlation'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1, 1])
        sns.lineplot(x='Time', y=CO2_f_norm, data=CO2_f, color='r', ax=axes[1, 1])
        axes[1, 1].set_title('BOLD vs Shifted CO2')
        axes[1, 1].legend(['BOLD', 'Shifted CO2'], facecolor='w')

        if verb:
            print('Saving CO2 plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'CO2.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        if verb:
            print('Creating stacked plot')
        
        f, axes = plt.subplots(2, 1)
        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0])
        sns.lineplot(x='Time', y=O2_norm, data=O2, color='g', ax=axes[0])
        sns.lineplot(x='Time', y=CO2_norm, data=CO2, color='r', ax=axes[0])
        axes[0].set_title('Processed Gases vs BOLD')
        axes[0].legend(['BOLD','Processed O2', 'Processed CO2'], facecolor='w')
        
        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1])
        sns.lineplot(x='Time', y=O2_f_norm, data=O2, color='g', ax=axes[1])
        sns.lineplot(x='Time', y=CO2_f_norm, data=CO2, color='r', ax=axes[1])
        axes[1].set_title('Shifted Gases vs BOLD')
        axes[1].legend(['BOLD','Shifted O2', 'Shifted CO2'], facecolor='w')

        if verb:
            print('Saving stacked plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'stacked.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        return    
        
    def save_plots(self, df, O2_time, O2, O2_shift, O2_correlation, O2_shift_f,
                   CO2_time, CO2, CO2_shift, CO2_correlation, CO2_shift_f, meants,
                   coeff, coeff_f, comb_corr,
                   f_path, key, verb, time_points, TR):
        """
        Create and saves plots for CO2 and O2 data

        Parameters:
            df: dataframe
                data structure that holds the data
            O2: array-like
                O2 data to be graphed
            O2_shift: array-like
                time-shifted O2 data to be graphed
            CO2: array-like
                CO2 data to be graphed
            CO2_shift: array-like
                time-shifted CO2 data to be graphed
            meants: array-like
                meants of BOLD data
            f_path: string
                the path of the file
            key: string
                the key for the type of type of analysis
            verb: boolean
                flag for verbose output
            verb: boolean
                flag for verbose output
            TR: float
                repitition time from json

        Returns:
            None
        """

        # set the size of the graphs
        sns.set(rc={'figure.figsize':(60,20)})
        plt.rc('legend', fontsize='x-large')
        plt.rc('xtick', labelsize='x-large')
        plt.rc('axes', titlesize='x-large')

        # normalize data
        meants_norm = sg.detrend(meants)
#        meants_norm = meants - meants.mean()
        meants_norm /= meants_norm.std()

#        df.CO2 = sg.detrend(df.CO2)
#        df.CO2 /= df.CO2.std()

        CO2_norm = sg.detrend(CO2)
#        CO2_norm = CO2 - CO2.mean()
        CO2_norm /= CO2_norm.std()

        CO2_shift_norm = sg.detrend(CO2_shift)
#        CO2_shift_norm = CO2_shift - CO2_shift.mean()
        CO2_shift_norm /= CO2_shift_norm.std()

#        df.O2 = sg.detrend(df.O2)
#        df.O2 /= df.O2.std()
        
        O2_norm = sg.detrend(O2)
#        O2_norm = O2 - O2.mean()
        O2_norm /= O2_norm.std()

        O2_shift_norm = sg.detrend(O2_shift)
#        O2_shift_norm = O2_shift - O2_shift.mean()
        O2_shift_norm /= O2_shift_norm.std()


        if verb:
            print('Creating O2 plots to be saved')
        # create subplots for png file later
        f, axes = plt.subplots(2, 2)
        
#        if df.Time.max() < 10:
#            df.Time = df.Time * 60

        sns.lineplot(x='Time', y='O2', data=df, linewidth=1, color='b', ax=axes[0, 0])
        sns.lineplot(x=O2_time, y=O2, linewidth=2, color='g', ax=axes[0, 0])
        axes[0,0].set_title('Processed O2 vs Raw')
        axes[0,0].legend(['Raw', 'Processed O2'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0,1])
        sns.lineplot(x=O2_time, y=O2_norm, color='g', ax=axes[0,1])
        axes[0,1].set_title('Processed O2 vs BOLD')
        axes[0,1].legend(['BOLD', 'Processed O2'], facecolor='w')
        
        sns.lineplot(x=np.arange(-len(O2_correlation)//2, len(O2_correlation)//2)+1, y=O2_correlation, ax=axes[1,0])
        axes[1,0].set_title('Cross-Correlation')
        axes[1,0].legend(['Cross-Correlation'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1,1])
        sns.lineplot(x=time_points, y=O2_shift_norm, color='g', ax=axes[1,1])
        axes[1,1].set_title('Shifted O2 vs BOLD')
        axes[1,1].legend(['BOLD', 'Shifted O2'], facecolor='w')

        # save the plot
        if verb:
            print('Saving O2 plots for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'O2_graph.png'
        f.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)

        if verb:
            print('Creating CO2 plots to be saved')
        # create subplots for png file later
        f, axes = plt.subplots(2, 2)

        sns.lineplot(x='Time', y='CO2', data=df, linewidth=1, color='b', ax=axes[0, 0])
        sns.lineplot(x=O2_time, y=CO2, linewidth=2, color='r', ax=axes[0, 0])
        axes[0,0].set_title('Processed CO2 vs Raw')
        axes[0,0].legend(['Raw', 'Processed CO2'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0,1])
        sns.lineplot(x=CO2_time, y=CO2_norm, color='r', ax=axes[0,1])
        axes[0,1].set_title('Processed CO2 vs BOLD')
        axes[0,1].legend(['BOLD', 'Processed CO2'], facecolor='w')
        
        sns.lineplot(x=np.arange(-len(CO2_correlation)//2, len(CO2_correlation)//2)+1, y=CO2_correlation, ax=axes[1,0])
        axes[1,0].set_title('Cross-Correlation')
        axes[1,0].legend(['Cross-Correlation'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1,1])
        sns.lineplot(x=time_points, y=CO2_shift_norm, color='r', ax=axes[1,1])
        axes[1,1].set_title('Shifted CO2 vs BOLD')
        axes[1,1].legend(['BOLD', 'Shifted CO2'], facecolor='w')

        # save the plot
        if verb:
            print('Saving CO2 plots for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'CO2_graph.png'
        f.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        if verb:
            print('Creating regression plot')
        
        sns.set(rc={'figure.figsize':(30,20)})
#        sns.lineplot(x=time_points, y=meants_norm, color='blue', ax=axes[0])
#        predict = coeff[0]*O2_shift + coeff[1]*CO2_shift + coeff[2]
#        predict_norm = sg.detrend(predict)
#        predict_norm /= predict_norm.std()
#        sns.lineplot(x=time_points, y=predict_norm, color='violet', ax=axes[0])
        
        f, axes = plt.subplots(3, 1)
        
        sns.lineplot(x=time_points, y=meants_norm, color='blue', ax=axes[0])
        predict = coeff[0]*O2_shift + coeff[1]*CO2_shift + coeff[2]
        predict_norm = sg.detrend(predict)
        predict_norm /= predict_norm.std()
        sns.lineplot(x=time_points, y=predict_norm, color='violet', ax=axes[0])
        axes[0].set_title('First-shift vs BOLD')
        axes[0].legend(['BOLD', 'First-shift'], facecolor='w')
        
        sns.lineplot(x=np.arange(-len(comb_corr)//2, len(comb_corr)//2)+1, y=comb_corr, ax=axes[1])
        axes[1].set_title('Cross-Correlation')
        axes[1].legend(['Cross-Correlation'], facecolor='w')

        sns.lineplot(x=time_points, y=meants_norm, color='blue', ax=axes[2])
        predict = coeff_f[0]*O2_shift_f + coeff_f[1]*CO2_shift_f + coeff_f[2]
        predict_norm = sg.detrend(predict)
        predict_norm /= predict_norm.std()
        sns.lineplot(x=time_points, y=predict_norm, color='violet', ax=axes[2])
        axes[2].set_title('Second-shift vs BOLD')
        axes[2].legend(['BOLD', 'Second-shift'], facecolor='w')

        if verb:
            print('Saving regression plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'regression.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)
        
        return
    
    def get_info(self, sigs, sig_fit):
        
#        for sig in sigs:
#            sig = sg.detrend(sig)
#            sig -= sig.mean()
#            sig /= sig.std()
        
#        sig_fit -= sig_fit.mean()
#        sig_fit = sg.savgol_filter(sig_fit, 11, 3)
#        sig_fit = sg.detrend(sig_fit)
#        sig_fit /= sig_fit.std()
#        sig_cut = []
#        
#        for sig in sigs:
#            sig_cut.append(sig[:len(sig_fit)])
#            
#        print(len(sig_cut[0]))
        X = np.vstack((np.array(sigs), np.ones_like(sigs[0]))).T
#        print(len(X))
        
#        clf = SGD(loss='huber', alpha=0.001, max_iter=2e9, tol=1e-6, learning_rate='optimal', shuffle=True, fit_intercept=True)
        clf = Ridge(alpha=0.001, max_iter=2e9, tol=1e-6, fit_intercept=True, normalize=False, solver='sag', copy_X=True)
        
#        diff = len(sigs[0]) - len(sig_fit)
#        sig_fit = pd.concat((sig_fit, pd.Series([0] * diff)))
        
        clf.fit(X, sig_fit)
#        print(clf.coef_)
        
#        n, k = X.shape
#        y_hat = np.matrix(clf.predict(X)).T
#        
#        # change X and sig_fit ot nump matricies. X aslo has a column of ones added
#        x = np.hstack((np.ones((n,1)), np.matrix(X)))
#        y = np.matrix(sig_fit).T
#        
#        # degrees of freedom
#        df = float(n-k-1)
#        
#        #sample var
#        sse = np.sum(np.square(y_hat - y), axis=0)
#        samp_var = sse/df
#        
#        # samp var for x
#        samp_var_x = x.T * x
#        
#        # covariant matrix
#        cov_mat = np.linalg.sqrtm(samp_var[0,0] * samp_var_x.I)
#        
#        # standard errors
#        se = cov_mat.diagonal()[1:]
#        
#        sse = np.sum((clf.predict(X) - sig_fit) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
#        print(sse)
#        print(sse.shape)
#        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
#        
#        t_value = clf.coef_ / se
#        p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), sig_fit.shape[0] - X.shape[1]))
        
        X = X.T
        stat = stats.pearsonr(clf.coef_[0] * X[0] + clf.coef_[1] * X[1] + clf.coef_[2] * X[2], sig_fit)
#        stat = stats.pearsonr(clf.coef_[0] * X[0] + clf.coef_[1] * X[1] + clf.coef_[2] * X[2] + clf.coef_[3] * X[3], sig_fit)
#        print(stat)
#        print()
        return clf.coef_, stat[0], stat[1]
        

#    def get_r2(self, sig_fit, sig_obs):
#        """
#        gets r^2 (coefficient of determination) of sig_fit w.r.t. sig_obs
#
#        inputs:
#            sig_fit (iterable) = fitted/predicted signal
#            sig_obs (iterable) = observed signal / raw data
#
#        warnings:
#            sig's must have the same lengths
#
#        return:
#            (float)
#        """
#        if(len(sig_fit) != len(sig_fit)):
#            print("Signals have different lengths: ", len(sig_fit) , ' &', len(sig_obs))
#        else:
#            sig_obs_var = np.mean(sig_obs)
#            return 1-np.sum((sig_fit-sig_obs)**2)/np.sum((sig_obs-sig_obs_var)**2)

    def resamp(self, og_time, new_time, data, shift, start):
        
        resample = interp.interp1d(og_time, data, fill_value='extrapolate')
        shifted = resample(new_time)
        
        if shift > 0:
            shifted[:start] = data[0]
            
        if shift < 0:
            shifted[start:] = data[len(data)-1]

#        shifted = sg.savgol_filter(shifted, 11, 3)
        
        return shifted
    
    def trim_edges(self, df):
        
        i = len(df)-1
        change = abs((df.Data[i-1] - df.Data[i]))
#        print(slope)
        count = 0
        
#        print(change)
        while(df.Time[i] > df.Time.max()*0.9):
            pre_change = change
            change = abs((df.Data[i-1] - df.Data[i]))
            diff = abs(change-pre_change)
#            print(diff)
            if diff > 5e-2:
                break
            count += 1
            i -= 1
        
        df.Data[len(df)-count:] = df.Data[len(df)-count]

        i = 0
        count = 0
        change = abs((df.Data[i+1] - df.Data[i]))
#        print(slope)
        
#        print(change)
        while(df.Time[i] < df.Time.max()*0.1):
            pre_change = change
            change = abs((df.Data[i+1] - df.Data[i]))
            diff = abs(change-pre_change)
#            print(diff)
            if diff > 5e-2:
                break
            count += 1
            i += 1
        
        df.Data[:count] = df.Data[count]
        
        return df
        

class peak_analysis:
    """docstring for peak_analysis."""

    def peak_four(self, df, verb, file, tr, time_pts, trough):
        
        f_O2 = fft_analysis().fourier_filter_no_resample(df.Time, df.O2, 2/60, 25/60)
        f_CO2 = fft_analysis().fourier_filter_no_resample(df.Time, df.CO2, 2/60, 25/60)
        
        resamp_tp = np.arange(0, df.Time.max(), tr)

        # get the troughs of the O2 data
        O2_data, _ = sg.find_peaks(df.O2.apply(lambda x:x*-1), prominence=2)
        O2_df = df.drop(columns=['CO2'])
        O2_df = O2_df.iloc[O2_data]
        
        f_O2_resamp = interp.interp1d(f_O2.Time, f_O2.Data, fill_value='extrapolate')
        f_O2_final = f_O2_resamp(O2_df.Time)
        
#        sns.lineplot(data=f_O2_final)
#        sns.lineplot(data=O2_df.O2)
#        plt.show()
        
        O2_df['cmp'] = O2_df.O2 < f_O2_final
        
        O2_valid_df = O2_df[O2_df.cmp == True]
        
#        O2_valid_df = O2_valid_df[O2_valid_df.Time > 5].reset_index(drop=True)
#        O2_valid_df = O2_valid_df[O2_valid_df.Time < O2_valid_df.Time.max()-5].reset_index(drop=True)
#        O2_valid_df.O2 = sg.savgol_filter(O2_valid_df.O2, 3, 2)
        
        O2_resamp = interp.interp1d(O2_valid_df.Time, O2_valid_df.O2, fill_value='extrapolate')
        O2_final_df = pd.DataFrame({'Time' : resamp_tp,
                                    'Data' : O2_resamp(resamp_tp)})
        
        O2_final_df = self._trim_edges(O2_final_df)
        
#        sns.lineplot(data=O2_final_df.Data)
#        plt.show()
        
        if trough:
            CO2_data, _ = sg.find_peaks(df.CO2.apply(lambda x:x*-1), prominence=3)
        else:
            CO2_data, _ = sg.find_peaks(df.CO2, prominence=3) 
        
        CO2_df = df.drop(columns=['O2'])
        CO2_df = CO2_df.iloc[CO2_data]
        
        f_CO2_resamp = interp.interp1d(f_CO2.Time, f_CO2.Data , fill_value='extrapolate')
        f_CO2_final = f_CO2_resamp(CO2_df.Time)
        
#        sns.lineplot(data=f_CO2_final)
#        sns.lineplot(data=CO2_df.CO2)
#        plt.show()
        
        if trough:
            CO2_df['cmp'] = CO2_df.CO2 < f_CO2_final
        else:
            CO2_df['cmp'] = CO2_df.CO2 > f_CO2_final
        
        CO2_valid_df = CO2_df[CO2_df.cmp == True].reset_index(drop=True)
#        CO2_valid_df = CO2_df
        
#        CO2_valid_df = CO2_valid_df[CO2_valid_df.Time > 5].reset_index(drop=True)
#        CO2_valid_df = CO2_valid_df[CO2_valid_df.Time < CO2_valid_df.Time.max()-5].reset_index(drop=True)
#        CO2_valid_df.CO2 = sg.savgol_filter(CO2_valid_df.CO2, 3, 2)
        
        CO2_resamp = interp.interp1d(CO2_valid_df.Time, CO2_valid_df.CO2, fill_value='extrapolate')
        CO2_final_df = pd.DataFrame({'Time' : resamp_tp,
                                     'Data' : CO2_resamp(resamp_tp)})
        
        CO2_final_df = self._trim_edges(CO2_final_df)
        
#        sns.lineplot(data=CO2_final_df.Data)
#        plt.show()
    
        return CO2_final_df, O2_final_df
    
    def peak(self, df, verb, file, time_points):
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

    def get_wlen(self, sig_time, sig):

        freq,_,power = fft_analysis().fourier_trans(max(sig_time)/len(sig_time), sig)
        window_it = np.argmax(power[25:500])+25
        freq_val = freq[window_it] # this is the most prominent frequency
        window_mag = 1/freq_val

        window_length = 0
        for i, t in enumerate(sig_time):
            if t > window_mag:
                window_length = i-1
                break

        return window_length


    def block_signal(self, sig_time, sig, time_pts):
        """
        Params:
            sigtime (iterable) = the sampling time points of sig
            sig (iterable) = signal to perform peakfinding and resampling on

        Return: (iterable)
            Peak found time-series which aligns with base_timeit
        """
        cpy = sig
        cpy = cpy.reset_index(drop=True)
        count = 0
        
#        if sig_time.max() < 10:
#            sig_time = sig_time * 60

        window_length = self.get_wlen(sig_time, cpy)

        for i in range(0,len(cpy),window_length):
            for j in range(i, i+window_length):
                if j < len(cpy):
                    cpy[j] = cpy[cpy[i:i+window_length].idxmax()]
            count += 1

        # get the sampling freq
        fs = len(cpy)/np.max(sig_time)
        # get cutoff freq
        fc = count / np.max(sig_time)
        w = fc / (fs / 2)

        b, a = sg.butter(4, w, 'low', analog=False)
        filtered = sg.filtfilt(b, a, cpy)

        return pd.DataFrame({'Time' : sig_time,
                             'Data' : filtered})
#        signal_resampler = interp.interp1d(sig_time, filtered, fill_value='extrapolate')
#        signal_resampled = signal_resampler(time_pts)
#
#        return signal_resampled

class shifter:
    """
    docstring for shifter.
    """
    def edge_match(self, base, sig, tr, time_pts):
            
        pt_shift = 0
#        print(base.Data.mean())
#        print(sig.Data.mean())
        direct = None
        prev_direct = None
        
        base_match = base[base.Data > base.Data.mean()].reset_index(drop=True)
        sig_match = sig[sig.Data > sig.Data.mean()].reset_index(drop=True)
        
        while True:
            
            start_diff = sig_match.Time[0] - base_match.Time[0]
            end_diff = sig_match.Time[len(sig_match)-1] - base_match.Time[len(base_match)-1]
            
#            print(sig_match.Time[0], base_match.Time[0], np.abs(start_diff))
#            print(sig_match.Time[len(sig_match)-1], base_match.Time[len(base_match)-1], np.abs(end_diff))
#            print()
#            time.sleep(3)
            
            if prev_direct and direct == -prev_direct:
                break
            
            if end_diff < 0 and start_diff < 0:
                prev_direct = direct
                direct = 1
                pt_shift += 1
                sig_match.Time += tr
            elif end_diff > 0 and start_diff > 0:
                prev_direct = direct
                direct = -1
                pt_shift -= 1
                sig_match.Time -= tr
            elif start_diff > end_diff:
                prev_direct = direct
                direct = 1
                pt_shift += 1
                sig_match.Time += tr
            elif end_diff > start_diff:
                prev_direct = direct
                direct = -1
                pt_shift -= 1
                sig_match.Time -= tr
            elif prev_direct and direct == -prev_direct:
                break
            else:
                break
        
        time_shift = pt_shift * tr
        
        resamp = interp.interp1d(sig.Time+time_shift, sig.Data, fill_value='extrapolate')
        shifted = resamp(time_pts)
        
        if pt_shift > 0:
            shifted[:pt_shift] = sig.Data[0]
        if pt_shift < 0:
            shifted[pt_shift:] = sig.Data[len(sig)-1]
#        
#        end = len(final) - len(time_points)
#        final = final[:-end]
#        
#        df = pd.DataFrame({ 'Time' : time_points,
#                            'Data' : final,
#                            'BOLD' : base})
        
        df = pd.DataFrame({ 'Time' : time_pts,
                            'Data' : shifted,
                            'BOLD' : base.Data})

#        shifted = sg.savgol_filter(shifted, 11, 3)
        
        return df, shifted, time_shift, pt_shift

    def get_cross_correlation(self, base, sig, scan_time, ref_shift):
        """
        Params:
            base (iterable) : reference signal (used to align signal)
            sig (iterable) : signal that needs alignment

        Returns:
            (float) : shift value of signal
        """
        correlation_series = sg.correlate(base, sig, mode='full')
#        print(ref_shift)
        if ref_shift != None:
            limit = int(60 * len(base)/scan_time)
            lim_corr = correlation_series[len(correlation_series)//2-limit+ref_shift : len(correlation_series)//2+limit+1+ref_shift]
            shift_index = np.argmax(lim_corr) - (len(lim_corr)//2) + ref_shift
        else:
            shift_index = np.argmax(correlation_series) - (len(correlation_series)//2)
        shift_value = scan_time/len(base) * shift_index
#        shift_value = 0
#        shift_index = 0
#        while shift_value < 5:
#            shift_index += 1
#            shift_value = scan_time/len(base) * shift_index
            
        return shift_value, shift_index, correlation_series

    def raw_align(self, base, raw_other, pre_other, scan_time, time_points, ref_shift=None):
#        plt.plot(raw_other.Data)
#        plt.show()
        raw_padded = self.pad_zeros(raw_other.Data)
        shift, start, corr = self.get_cross_correlation(base, raw_padded, scan_time, ref_shift)
        resamp = interp.interp1d(pre_other.Time+shift, pre_other.Data, fill_value='extrapolate')
        shifted = resamp(time_points)
        
        if shift > 0:
            shifted[:start] = pre_other.Data.iloc[0]
        if shift < 0:
            shifted[start:] = pre_other.Data.iloc[len(pre_other)-1]
            
        df = pd.DataFrame({ 'Time' : time_points,
                            'Data' : shifted})
        
        return df, shifted, corr, shift, start
        
    def corr_align(self, base, other_time, other_sig, scan_time, time_points, ref_shift=None):
        '''
        Parameters:
            base: numpy array
                The base signal that the other signal will be shifted to matching
            other: numpy array
                The signal to be shifted

        Returns:
            shifted: numpy array
                The shifted other signal
        '''
#        shift, start, corr = self.get_cross_correlation(base, other_sig, scan_time)
#        base_norm = base - base.mean()
#        base_norm /= base_norm.std()

#        other_norm = other_sig - other_sig.mean()
#        other_norm /= other_norm.std()
        
#        print(len(base_norm))
#        print(len(other_norm))
        
#        _, gd= sg.group_delay((base_norm, other_norm))
#        print(gd)
#        print(len(gd))
        
#        other_norm= sg.savgol_filter(other_norm, 35, 3)
        
#        plt.plot(other_norm)
#        plt.show()
#        
#        other_sep = len(other_norm)//4
#        left_edge = np.argmax(other_norm[:other_sep])
##        print(left_edge)
#        right_edge = np.argmax(other_norm[other_sep*3:])+other_sep*3
##        print(right_edge)
#        other_norm = other_norm[left_edge:right_edge]
#        
#        base_sep = len(base_norm)//4
#        left_edge = np.argmax(base_norm[:base_sep])
##        print(left_edge)
#        right_edge = np.argmax(base_norm[base_sep*3:])+base_sep*3
##        print(right_edge)
#        base_norm = base_norm[left_edge:right_edge]
        
#        plt.plot(other_norm)
#        plt.show()

        #get shifts
#        print(len(other_sig))
#        print(len(base))
        
        base_norm = sg.detrend(base)
        other_norm = sg.detrend(other_sig)
#        shift, start, corr = self.get_cross_correlation(base_norm, other_norm, scan_time)
        other_padded = self.pad_zeros(other_norm)
#        print(ref_shift)
        shift, start, corr = self.get_cross_correlation(base_norm, other_padded, scan_time, ref_shift)
#        print(corr)
#        print(corr[len(corr)//2])
        #construct resampler
#        print(shift, start)
#        exit()
        resamp = interp.interp1d(other_time+shift, other_sig, fill_value='extrapolate')
        shifted = resamp(time_points)
        
        if shift > 0:
            shifted[:start] = other_sig[0]
#            prepend = [other_sig.iloc[0]] * start
#            prepend.append(other_sig)
#            final = prepend[:-start]
        if shift < 0:
            shifted[start:] = other_sig[len(other_sig)-1]
#            other_sig = other_sig[-start:]
#            print(-start)
#            append = [other_sig.iloc[-1]] * -start
#            final = other_sig.append(append)
#        
#        end = len(final) - len(time_points)
#        final = final[:-end]
#        
        df = pd.DataFrame({ 'Time' : time_points,
                            'Data' : shifted})
        
#        df = pd.DataFrame({ 'Time' : time_points,
#                            'Data' : shifted,
#                            'BOLD' : base})

#        shifted = sg.savgol_filter(shifted, 11, 3)
        
        return df, shifted, corr, shift, start
#        return df, df.Data, corr
        
    def pad_zeros(self, sig):
        pre = np.zeros_like(sig)
        post = np.zeros_like(sig)
        padded = np.concatenate((pre, sig, post))
        return padded

class optimizer:
    """docstring for optimizer."""

    def __grad_constant_GLM(self, c1,c2,c3,s1n, s2n, bn):
        """
        part of mathematical gradient which is present in all components

        params:
            c1 (float) = constant 1
            c2 (float) = constant 2
            c3 (float) = constant 3
            s1n (float) = signal1 at time n
            s2n (float) = signal2 at time n
            bn (float) = base signal at time n

        returns:
            (float)
        """
        return(2*(c1*s1n + c2*s2n + c3 - bn))
    def __grad_C1_GLM(self, C1, C2, C3, S1, S2, B):
        """
        component 1 of gradient

        params:
            C1 (float) = constant 1
            C2 (float) = constant 2
            C3 (float) = constant 3
            S1 (iterable) = signal 1
            S2 (iterable) = signal 2
            B (iterable) = base signal

        returns:
            (float)
        """
        buffer = 0.0
        for s1n, s2n, bn in zip(S1, S2, B):
            buffer += self.__grad_constant_GLM(C1, C2, C3, s1n, s2n, bn) * s1n/len(S1)
        return buffer

    def __grad_C2_GLM(self, C1, C2, C3, S1, S2, B):
        """
        component 2 of gradient

        params:
            C1 (float) = constant 1
            C2 (float) = constant 2
            C3 (float) = constant 3
            S1 (iterable) = signal 1
            S2 (iterable) = signal 2
            B (iterable) = base signal

        returns:
            (float)
        """
        buffer = 0.0
        for s1n, s2n, bn in zip(S1, S2, B):
            buffer += self.__grad_constant_GLM(C1, C2, C3, s1n, s2n, bn) * s2n/len(S1)
        return buffer

    def __grad_C3_GLM(self, C1, C2, C3, S1, S2, B):
        """
        component 3 of gradient

        params:
            C1 (float) = constant 1
            C2 (float) = constant 2
            C3 (float) = constant 3
            S1 (iterable) = signal 1
            S2 (iterable) = signal 2
            B (iterable) = base signal

        returns:
            (float)
        """
        buffer = 0.0
        for s1n, s2n, bn in zip(S1, S2, B):
            buffer += self.__grad_constant_GLM(C1, C2, C3, s1n, s2n, bn)/len(S1)
        return buffer

    def linear_optimize_GLM(self, S1, S2, B, init_tuple = (0,0,0), descent_speed=.1, lifespan = 10000):
        """
        linear coefficient optimizer by gradient descent

            params:
                S1 (iterable) = signal 1 (O2/CO2)
                S2 (iterable) = signal 2 (CO2/O2)
                B (iterable) = base signal (BOLD)
                init_tuple (c1, c2, c3) = initial coefficeint guess
                descent_speed (float) = descending step size
                lifespan = number of descending steps

        Warnings:
            non-stochastic gradient descent is extremely sensitive to local minima

        returns:
            (C1 (float), C2 (float), C3 (float)) where: C1 * S1 + C2 * S2 + C3 = B
        """
        factor1 = np.max(S1)
        factor2 = np.max(S2)
        factorB = np.max(B)

        S1 = S1/factor1
        S2 = S2/factor2
        B = B/factorB

        curr_C1 = init_tuple[0]*factor1/factorB
        curr_C2 = init_tuple[1]*factor2/factorB
        curr_C3 = init_tuple[2]/factorB


        for i in tqdm(range(lifespan)):
            curr_C1 = curr_C1 - (descent_speed* self.__grad_C1_GLM(curr_C1, curr_C2, curr_C3, S1, S2, B))
            curr_C2 = curr_C2 - (descent_speed* self.__grad_C2_GLM(curr_C1, curr_C2, curr_C3, S1, S2, B))
            curr_C3 = curr_C3 - (descent_speed* self.__grad_C3_GLM(curr_C1, curr_C2, curr_C3, S1, S2, B))

        return_tuple = (curr_C1*factorB/factor1, curr_C2*factorB/factor2, curr_C3*factorB)

        S1 = S1*factor1
        S2 = S2*factor2
        B = B*factorB
        return return_tuple

    def stochastic_optimize_GLM(self, S1, S2, B, init_tuple = (0,0,0), descent_speed=.1, lifespan = 10000, p_factor = .9):
        """
        linear coefficient optimizer by stochastic gradient descent

        params:
            S1 (iterable) = signal 1 (O2/CO2)
            S2 (iterable) = signal 2 (CO2/O2)
            B (iterable) = base signal (BOLD)
            init_tuple (c1, c2, c3) = initial coefficeint guess
            descent_speed (float) = descending step size
            lifespan = number of descending steps
            p_factor = momentum constant (determines the contribution of momentum and slope)

        returns:
            (C1 (float), C2 (float), C3 (float)) where: C1 * S1 + C2 * S2 + C3 = B
        """
        factor1 = np.max(S1)
        factor2 = np.max(S2)
        factorB = np.max(B)

        S1 = S1/factor1
        S2 = S2/factor2
        B = B/factorB

        curr_C1 = init_tuple[0]*factor1/factorB
        curr_C2 = init_tuple[1]*factor2/factorB
        curr_C3 = init_tuple[2]/factorB

        pC1 = 0.0
        pC2 = 0.0
        pC3 = 0.0

        for i in tqdm(range(lifespan)):
            pC1 = p_factor*pC1 + (1.0-p_factor)*self.__grad_C1_GLM(curr_C1, curr_C2, curr_C3, S1, S2, B)
            pC2 = p_factor*pC2 + (1.0-p_factor)*self.__grad_C2_GLM(curr_C1, curr_C2, curr_C3, S1, S2, B)
            pC3 = p_factor*pC3 + (1.0-p_factor)*self.__grad_C3_GLM(curr_C1, curr_C2, curr_C3, S1, S2, B)

            curr_C1 = curr_C1 - (descent_speed* pC1)
            curr_C2 = curr_C2 - (descent_speed* pC2)
            curr_C3 = curr_C3 - (descent_speed* pC3)

        return_tuple = (curr_C1*factorB/factor1, curr_C2*factorB/factor2, curr_C3*factorB)

        S1 = S1*factor1
        S2 = S2*factor2
        B = B*factorB
        return return_tuple

class parallel_processing:
    
    def kill_unending(self, processes, verb):
        proc = subprocess.Popen("ps -e | grep flirt", encoding='utf-8', stdout=subprocess.PIPE, shell=True)

        outs = proc.communicate()[0].split('\n')
        for line in outs:
            parts = line.split()
            if len(parts) > 0:
                time = parts[2].split(':')
                secs = int(time[0])*3600 + int(time[1])*60 + int(time[2])
                if secs > 900:
                    os.kill(int(parts[0]), signal.SIGTERM)
        
        return
    
    def get_next_avail(self, processes, verb, limit, key, s_name):
        msg = False
        spin = '|/-\\'
        cursor = 0
        while not any(v is None for v in processes):
            if verb:
                if not msg:
                    print('There are', limit, key, s_name, 'currently running. Limit reached. Waiting for at least one to end.')
                    msg = True
                else:
                    sys.stdout.write(spin[cursor])
                    sys.stdout.flush()
                    cursor += 1
                    if cursor >= len(spin):
                        cursor = 0
            
            self.kill_unending(processes, verb)
            
            for i, process in enumerate(processes):
                if process != None and process.poll() != None:
                    processes[i] = None
                    break
                    
            if verb:
                if msg:
                    time.sleep(0.2)
                    sys.stdout.write('\b')
        
        return processes.index(None)
    
    def wait_remaining(self, processes, verb, key, s_name):
        msg = False
        spin = '|/-\\'
        cursor = 0
            
        while not all(v is None for v in processes):
            if verb:
                if not msg:
                    print('Waiting for the remaining', key, s_name, 'to finish')
                    msg = True
                else:
                    sys.stdout.write(spin[cursor])
                    sys.stdout.flush()
                    cursor += 1
                    if cursor >= len(spin):
                        cursor = 0
            
            self.kill_unending(processes, verb)
            
            for i, process in enumerate(processes):
                if process != None and process.poll() != None:
                    processes[i] = None
                        
            if verb:
                if msg:
                    time.sleep(0.2)
                    sys.stdout.write('\b')
        
        return