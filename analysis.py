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
            df = stat_utils().trim_edges(df)
        
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
            df = stat_utils().trim_edges(df)
        
        return df

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
        sns.set(rc={'figure.figsize':(60,40)})
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
                        f_path, key, verb, time_points, disp):
        
        if verb:
            print('Creating O2 plots')
        
        # set the size of the graphs
        sns.set(rc={'figure.figsize':(30,20)})
        plt.rc('legend', fontsize='medium')
        plt.rc('xtick', labelsize='medium')
        plt.rc('ytick', labelsize='x-small')
        plt.rc('axes', titlesize='medium')
        if disp:
            plt.ion()
            plt.show()

        meants_norm = meants / meants.std()
        O2_norm = O2.Data / O2.Data.std()
        O2_m_norm = O2_m.Data / O2_m.Data.std()
        
        f, axes = plt.subplots(2, 2)
        f.suptitle(f_path[:-4].split('/')[-1])
        
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
        if disp:
            plt.draw()
            plt.pause(0.001)
        else:
            f.clf()
            plt.close(fig=f)
        
        if verb:
            print('Creating CO2 plots')
            
        f, axes = plt.subplots(2, 2)
        f.suptitle(f_path[:-4].split('/')[-1])
        
        CO2_norm = CO2.Data / CO2.Data.std()
        CO2_m_norm = CO2_m.Data / CO2_m.Data.std()
        
        sns.lineplot(x='Time', y='CO2', data=df, color='b', ax=axes[0, 0])
        sns.lineplot(x='Time', y='Data', data=CO2, color='r', ax=axes[0, 0])
        axes[0, 0].set_title('Raw CO2 vs Processed CO2')
        axes[0, 0].legend(['Raw CO2', 'Processed CO2'], facecolor='w')

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
        if disp:
            plt.draw()
            plt.pause(0.001)
        else:
            f.clf()
            plt.close(fig=f)
        
        if verb:
            print('Creating combined plots')
            
        f, axes = plt.subplots(3, 1)
        f.suptitle(f_path[:-4].split('/')[-1])
        
        combined = coeff[0] * O2_m_norm + coeff[1] * CO2_m_norm + coeff[2]
        combined /= combined.std()
        
        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0])
        sns.lineplot(x='Time', y=combined, data=CO2, color='black', ax=axes[0])
        axes[0].set_title('BOLD vs Combination')
        axes[0].legend(['BOLD', 'Combination'], facecolor='w')
        
        sns.lineplot(x=np.arange(-len(comb_corr)//2, len(comb_corr)//2)+1, y=comb_corr, ax=axes[1])
        axes[1].set_title('Cross-Correlation')
        axes[1].legend(['Cross-Correlation'], facecolor='w')

        
        O2_f_norm = O2_f / O2_f.std()
        CO2_f_norm = CO2_f / CO2_f.std()
        combined_s = coeff_f[0] * O2_f_norm + coeff_f[1] * CO2_f_norm + coeff_f[2]
        combined_s /= combined_s.std()
        
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
        if disp:
            plt.draw()
            plt.pause(0.001)
        else:
            f.clf()
            plt.close(fig=f)
        
        if verb:
            print('Creating stacked plot')
        
        f, axes = plt.subplots(2, 1)
        f.suptitle(f_path[:-4].split('/')[-1])
        
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
        if disp:
            plt.draw()
            plt.pause(0.001)
        else:
            f.clf()
            plt.close(fig=f)
        
        return    
    
    def save_plots_comb_extend(self, df, O2, O2_m, O2_f, O2_corr,
                               CO2, CO2_m, CO2_f, CO2_corr, meants,
                               coeff, coeff_f, comb_corr, extend_time,
                               f_path, key, verb, time_points, disp):
        
        if verb:
            print('Creating O2 plots')
        
        # set the size of the graphs
        sns.set(rc={'figure.figsize':(30,20)})
        plt.rc('legend', fontsize='medium')
        plt.rc('xtick', labelsize='medium')
        plt.rc('ytick', labelsize='x-small')
        plt.rc('axes', titlesize='medium')
        if disp:
            plt.ion()
            plt.show()

        meants_norm = meants / meants.std()
        O2_norm = O2.Data / O2.Data.std()
        O2_m_norm = O2_m.Data / O2_m.Data.std()
        
        f, axes = plt.subplots(2, 2)
        f.suptitle(f_path[:-4].split('/')[-1])
        
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
        if disp:
            plt.draw()
            plt.pause(0.001)
        else:
            f.clf()
            plt.close(fig=f)
        
        if verb:
            print('Creating CO2 plots')
            
        f, axes = plt.subplots(2, 2)
        f.suptitle(f_path[:-4].split('/')[-1])
        
        CO2_norm = CO2.Data / CO2.Data.std()
        CO2_m_norm = CO2_m.Data / CO2_m.Data.std()
        
        sns.lineplot(x='Time', y='CO2', data=df, color='b', ax=axes[0, 0])
        sns.lineplot(x='Time', y='Data', data=CO2, color='r', ax=axes[0, 0])
        axes[0, 0].set_title('Raw CO2 vs Processed CO2')
        axes[0, 0].legend(['Raw CO2', 'Processed CO2'], facecolor='w')

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
        if disp:
            plt.draw()
            plt.pause(0.001)
        else:
            f.clf()
            plt.close(fig=f)
        
        if verb:
            print('Creating combined plots')
            
        f, axes = plt.subplots(3, 1)
        f.suptitle(f_path[:-4].split('/')[-1])
        
        combined = coeff[0] * O2_m_norm + coeff[1] * CO2_m_norm + coeff[2]
        combined /= combined.std()
        
        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0])
        sns.lineplot(x='Time', y=combined, data=CO2, color='black', ax=axes[0])
        axes[0].set_title('BOLD vs Combination')
        axes[0].legend(['BOLD', 'Combination'], facecolor='w')
        
        sns.lineplot(x=np.arange(-len(comb_corr)//2, len(comb_corr)//2)+1, y=comb_corr, ax=axes[1])
        axes[1].set_title('Cross-Correlation')
        axes[1].legend(['Cross-Correlation'], facecolor='w')

        
        O2_f_norm = O2_f / O2_f.std()
        CO2_f_norm = CO2_f / CO2_f.std()
        combined_s = coeff_f[0] * O2_f_norm + coeff_f[1] * CO2_f_norm + coeff_f[2]
        combined_s /= combined_s.std()
        
        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[2])
        sns.lineplot(x=extend_time, y=combined_s, color='black', ax=axes[2])
        axes[2].set_title('BOLD vs Shifted Combination')
        axes[2].legend(['BOLD', 'Shifted Combination'], facecolor='w')
        
        if verb:
            print('Saving combined plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'combined.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
        if disp:
            plt.draw()
            plt.pause(0.001)
        else:
            f.clf()
            plt.close(fig=f)
        
        if verb:
            print('Creating stacked plot')
        
        f, axes = plt.subplots(2, 1)
        f.suptitle(f_path[:-4].split('/')[-1])
        
        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0])
        sns.lineplot(x='Time', y=O2_m_norm, data=O2_m, color='g', ax=axes[0])
        sns.lineplot(x='Time', y=CO2_m_norm, data=CO2_m, color='r', ax=axes[0])
        sns.lineplot(x=time_points, y=combined, color='black', ax=axes[0])
        axes[0].set_title('Processed Gases vs BOLD')
        axes[0].legend(['BOLD','Processed O2', 'Processed CO2', 'Combination'], facecolor='w')
        
        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1])
        sns.lineplot(x=extend_time, y=O2_f_norm, color='g', ax=axes[1])
        sns.lineplot(x=extend_time, y=CO2_f_norm, color='r', ax=axes[1])
        sns.lineplot(x=extend_time, y=combined_s, color='black', ax=axes[1])
        axes[1].set_title('Shifted Gases vs BOLD')
        axes[1].legend(['BOLD','Shifted O2', 'Shifted CO2', 'Shifted Combination'], facecolor='w')

        if verb:
            print('Saving stacked plot for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'stacked.png'
        plt.savefig(save_path)
        if verb:
            print('Saving complete')
        if disp:
            plt.draw()
            plt.pause(0.001)
        else:
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
        '''
        Get the coefficents to preform a linear combination of the 2 sigs to fit to sig_fit
        
        Parameters:
            sigs : a 2-d array [x, y]
                a 2-d array that contains the data_points for the 2 signals
            sig_fit :
                an array that contains the target signal to be fit
        
        Notes:
            the lenght of the signals in sigs must be the same length as the sig_fit signal
            
        Returns:
            clf.coef_ : an array of coefficents
            r_value : the r_value of the combination vs sig_fit
            p_value : the p_value of the combination vs sig_fit
        '''
        
        # transpose the signals add a column of 1s to account for constant
        X = np.vstack((np.array(sigs), np.ones_like(sigs[0]))).T
#        print(X)
#        print(X.shape)
#        print(type(X))
        X_contig = np.ascontiguousarray(X)
        # use stocastic average gradent decent to caculate the coefficents
        clf = Ridge(alpha=0.001, max_iter=2e9, tol=1e-6, fit_intercept=True, normalize=False, solver='sag', copy_X=True)
        # calculate the coefficents
        clf.fit(X_contig, sig_fit)
        # return X back to its original dimensions
        X = X.T
        # calculate the stats for combined signal
        stat = stats.pearsonr(clf.coef_[0] * X[0] + clf.coef_[1] * X[1] + clf.coef_[2] * X[2], sig_fit)

        return clf.coef_, stat[0], stat[1]

    def resamp(self, og_time, new_time, data, shift, start):
        '''
        Interpolates the data from original time to the new target time
        
        Parameters: 
            og_time : pandas Series or array-like
                Time of the original data series
            new_time : pandas Sereis or array-like
                New time that is to be interpolated to
            data : pandas Series or array-like
                Data that is to be interpoalted
            shift : float
                The time shift between the new time and old time
            start : int
                The number of data points to shift between the new time and old time
        
        Returns:
            shifted : 1-d numpy array of the time shifted and interpolated data
        '''
        
        resample = interp.interp1d(og_time, data, fill_value='extrapolate')
        shifted = resample(new_time)
        
        if shift > 0:
            shifted[:start] = data[0]
            
        if shift < 0:
            extra = og_time.max() - new_time.max()
            ex_pts = int(extra // new_time[1])
            if ex_pts > 0:
                safe = ex_pts + start
                if safe < 0:
                    shifted[safe:] = data[len(data)-1]
            else:
                shifted[len(data)+start:] = data[len(data)-1]
        
        return shifted
    
    def resamp_f(self, og_time, new_time, data, shift, start, tr):
        '''
        Interpolates the data from original time to the new target time
        
        Parameters: 
            og_time : pandas Series or array-like
                Time of the original data series
            new_time : pandas Sereis or array-like
                New time that is to be interpolated to
            data : pandas Series or array-like
                Data that is to be interpoalted
            shift : float
                The time shift between the new time and old time
            start : int
                The number of data points to shift between the new time and old time
        
        Returns:
            shifted : 1-d numpy array of the time shifted and interpolated data
        '''
        last_time = new_time[-1]
        to_add = [last_time+tr, last_time+2*tr, last_time+3*tr, last_time+4*tr, last_time+5*tr, last_time+6*tr, last_time+7*tr]
        new_time = np.append(new_time, to_add)
        resample = interp.interp1d(og_time, data, fill_value='extrapolate')
        shifted = resample(new_time)
        
        if shift > 0:
            shifted[:start] = data[0]
            
        if shift < 0:
            extra = og_time.max() - new_time.max()
            ex_pts = int(extra // new_time[1])
            if ex_pts > 0:
                safe = ex_pts + start
                if safe < 0:
                    shifted[safe:] = data[len(data)-1]
            else:
                shifted[len(data)+start:] = data[len(data)-1]
        
        return shifted, new_time
    
    def trim_edges(self, df):
        '''
        Trims to edges of the data that are considered noise
        
        Parameters:
            df : pandas Dataframe
                Data that is to be trimmed
        
        Returns:
            df : pandas Dataframe with trimmed data
        '''
        i = len(df)-1
        change = abs((df.Data[i-1] - df.Data[i]))
        count = 0
        
        while(df.Time[i] > df.Time.max()*0.9):
            pre_change = change
            change = abs((df.Data[i-1] - df.Data[i]))
            diff = abs(change-pre_change)
            if diff > 5e-2:
                break
            count += 1
            i -= 1
        
        df.Data[len(df)-count:] = df.Data[len(df)-count]

        i = 0
        count = 0
        change = abs((df.Data[i+1] - df.Data[i]))
        
        while(df.Time[i] < df.Time.max()*0.1):
            pre_change = change
            change = abs((df.Data[i+1] - df.Data[i]))
            diff = abs(change-pre_change)
            if diff > 5e-2:
                break
            count += 1
            i += 1
        
        df.Data[:count] = df.Data[count]
        
        return df
        

class peak_analysis:
    """docstring for peak_analysis."""

    def peak_four(self, df, verb, file, tr, time_pts, trough):
        
#        f_O2 = fft_analysis().fourier_filter_no_resample(df.Time, df.O2, 1/60, tr, trim=True)
        f_CO2 = fft_analysis().fourier_filter_no_resample(df.Time, df.CO2, 1/60, tr, trim=True)
        
        resamp_tp = np.arange(0, df.Time.max(), tr)

        # get the troughs of the O2 data
        O2_data, _ = sg.find_peaks(df.O2.apply(lambda x:x*-1), prominence=2)
        O2_df = df.drop(columns=['CO2'])
        O2_df = O2_df.iloc[O2_data]
        
#        f_O2_resamp = interp.interp1d(f_O2.Time, f_O2.Data, fill_value='extrapolate')
#        f_O2_final = f_O2_resamp(O2_df.Time)
        
#        sns.lineplot(x=df.Time, y=df.O2)
#        sns.lineplot(x=O2_df.Time, y=f_O2_final)
#        sns.lineplot(x=O2_df.Time, y=O2_df.O2)
#        plt.show()
        
#        O2_df['cmp'] = O2_df.O2 < f_O2_final
#        
#        O2_valid_df = O2_df[O2_df.cmp == True]
        O2_valid_df = O2_df
        
#        O2_valid_df = O2_valid_df[O2_valid_df.Time > 5].reset_index(drop=True)
#        O2_valid_df = O2_valid_df[O2_valid_df.Time < O2_valid_df.Time.max()-5].reset_index(drop=True)
#        O2_valid_df.O2 = sg.savgol_filter(O2_valid_df.O2, 3, 2)
        
        O2_resamp = interp.interp1d(O2_valid_df.Time, O2_valid_df.O2, fill_value='extrapolate')
        O2_final_df = pd.DataFrame({'Time' : resamp_tp,
                                    'Data' : O2_resamp(resamp_tp)})
        
        O2_final_df = stat_utils().trim_edges(O2_final_df)
        
#        sns.lineplot(data=O2_final_df.Data)
#        plt.show()
        
        if trough:
            CO2_data, _ = sg.find_peaks(df.CO2.apply(lambda x:x*-1), prominence=3, width=20)
        else:
            CO2_data, _ = sg.find_peaks(df.CO2, prominence=3, width=30) 
        
        CO2_df = df.drop(columns=['O2'])
        CO2_df = CO2_df.iloc[CO2_data]
        
        f_CO2_resamp = interp.interp1d(f_CO2.Time, f_CO2.Data , fill_value='extrapolate')
        f_CO2_final = f_CO2_resamp(CO2_df.Time)
        
#        sns.lineplot(x=df.Time, y=df.CO2)
#        sns.lineplot(x=CO2_df.Time, y=f_CO2_final)
#        sns.lineplot(x=CO2_df.Time, y=CO2_df.CO2)
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
        
        CO2_final_df = stat_utils().trim_edges(CO2_final_df)
        
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
#            plt.show()
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
#            plt.show()
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

        freq,_,disp = fft_analysis().fourier_trans(sig_time, sig, len(sig))
#        plt.plot(sig)
#        plt.show()
#        plt.plot(disp)
#        plt.show()
#        plt.plot(disp[25:500])
#        plt.show()
#        exit()
        window_it = np.argmax(disp[25:500])+25
        freq_val = freq[window_it] # this is the most prominent frequency
        window_mag = 1/freq_val

        window_length = 0
        for i, t in enumerate(sig_time):
            if t > window_mag:
                window_length = i-1
                break

        return window_length


    def envelope(self, sig_time, sig, tr, invert):
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
#        cpy = []
#        time = []
        count = 0
        
#        if sig_time.max() < 10:
#            sig_time = sig_time * 60

        window_length = self.get_wlen(sig_time, cpy)
#        plt.plot(sig)
#        plt.show()
#        plt.plot(cpy)
#        plt.show()
        for i in range(0,len(cpy),window_length):
            for j in range(i, i+window_length):
                if j < len(cpy):
                    cpy[j] = cpy[cpy[i:i+window_length].idxmax()]
#                    index = sig[i:i+window_length].idxmax()
#                    cpy.append(sig[index])
#                    time.append(sig_time[index])
            count += 1
        if invert:
            cpy *= -1
#        plt.plot(cpy)
#        plt.show()
        # get the sampling freq
        fs = len(cpy)/np.max(sig_time)
        # get cutoff freq
        fc = count / np.max(sig_time)
        w = fc / (fs / 2)

        b, a = sg.butter(6, w, 'low', analog=False)
        filtered = sg.filtfilt(b, a, cpy)
#        plt.plot(filtered)
#        plt.show()
#        return pd.DataFrame({'Time' : sig_time,
#                             'Data' : filtered})
#        signal_resampler = interp.interp1d(sig_time, filtered, fill_value='extrapolate')
        signal_resampled = stat_utils().resamp(sig_time, time_pts, filtered, 0, 0)
#        plt.plot(signal_resampled)
#        plt.show()
        return pd.DataFrame({'Time' : time_pts,
                             'Data' : signal_resampled})
#
#        return signal_resampled

class shifter:
    """
    docstring for shifter.
    """
#    def edge_match(self, base, sig, tr, time_pts):
#            
#        pt_shift = 0
##        print(base.Data.mean())
##        print(sig.Data.mean())
#        direct = None
#        prev_direct = None
#        
#        base_match = base[base.Data > base.Data.mean()].reset_index(drop=True)
#        sig_match = sig[sig.Data > sig.Data.mean()].reset_index(drop=True)
#        
#        while True:
#            
#            start_diff = sig_match.Time[0] - base_match.Time[0]
#            end_diff = sig_match.Time[len(sig_match)-1] - base_match.Time[len(base_match)-1]
#            
##            print(sig_match.Time[0], base_match.Time[0], np.abs(start_diff))
##            print(sig_match.Time[len(sig_match)-1], base_match.Time[len(base_match)-1], np.abs(end_diff))
##            print()
##            time.sleep(3)
#            
#            if prev_direct and direct == -prev_direct:
#                break
#            
#            if end_diff < 0 and start_diff < 0:
#                prev_direct = direct
#                direct = 1
#                pt_shift += 1
#                sig_match.Time += tr
#            elif end_diff > 0 and start_diff > 0:
#                prev_direct = direct
#                direct = -1
#                pt_shift -= 1
#                sig_match.Time -= tr
#            elif start_diff > end_diff:
#                prev_direct = direct
#                direct = 1
#                pt_shift += 1
#                sig_match.Time += tr
#            elif end_diff > start_diff:
#                prev_direct = direct
#                direct = -1
#                pt_shift -= 1
#                sig_match.Time -= tr
#            elif prev_direct and direct == -prev_direct:
#                break
#            else:
#                break
#        
#        time_shift = pt_shift * tr
#        
#        resamp = interp.interp1d(sig.Time+time_shift, sig.Data, fill_value='extrapolate')
#        shifted = resamp(time_pts)
#        
#        if pt_shift > 0:
#            shifted[:pt_shift] = sig.Data[0]
#        if pt_shift < 0:
#            shifted[pt_shift:] = sig.Data[len(sig)-1]
##        
##        end = len(final) - len(time_points)
##        final = final[:-end]
##        
##        df = pd.DataFrame({ 'Time' : time_points,
##                            'Data' : final,
##                            'BOLD' : base})
#        
#        df = pd.DataFrame({ 'Time' : time_pts,
#                            'Data' : shifted,
#                            'BOLD' : base.Data})
#
##        shifted = sg.savgol_filter(shifted, 11, 3)
#        
#        return df, shifted, time_shift, pt_shift

    def get_cross_correlation(self, base, sig, scan_time, ref_shift, invert):
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
        if invert:
            shift_index = 0
        elif ref_shift != None:
            limit = int(30 * len(base)/scan_time)
            lim_corr = correlation_series[len(correlation_series)//2-limit+ref_shift : len(correlation_series)//2+limit+1+ref_shift]
#            plt.plot(lim_corr)
#            plt.show()
            shift_index = np.argmax(lim_corr) - (len(lim_corr)//2) + ref_shift
        else:
#            plt.plot(correlation_series)
#            plt.show()
            shift_index = np.argmax(correlation_series) - (len(correlation_series)//2)
        shift_value = scan_time/len(base) * shift_index
            
        return shift_value, shift_index, correlation_series

    def raw_align(self, base, raw_other, scan_time, time_points, ref_shift, invert):
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
        raw_padded = self.pad_zeros(raw_other.Data)
        shift, start, corr = self.get_cross_correlation(base, raw_padded, scan_time, ref_shift, invert)
        shifted = stat_utils().resamp(raw_other.Time+shift, raw_other.Time, raw_other.Data, shift, start)
            
        df = pd.DataFrame({ 'Time' : time_points,
                            'Data' : shifted})
        
        return df, shifted, corr, shift, start
        
    def corr_align(self, base, other_time, other_sig, scan_time, time_points, ref_shift, invert):
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

#        other_padded = self.pad_zeros(other_sig)
        shift, start, corr = self.get_cross_correlation(base, other_sig, scan_time, ref_shift, invert)
        
        shifted = stat_utils().resamp(other_time+shift, time_points, other_sig, shift, start)
        
        df = pd.DataFrame({ 'Time' : time_points,
                            'Data' : shifted})
        
        return df, shifted, corr, shift, start
        
    def pad_zeros(self, sig):
        pre = np.zeros_like(sig)
        post = np.zeros_like(sig)
        padded = np.concatenate((pre, sig, post))
        return padded


class parallel_processing:
    '''
    This class is to allow there to be processing multiple FEATs at once
    '''
    
    def kill_unending(self, processes, verb):
        '''
        Kills processes if flirt has been running for too long so the FEAT can continue
        
        Parameters:
            processes : array-like
                List of processes that are being run
            verb : boolean
                Flag to turn on verbose output
                
        Returns:
            Nothing
        '''
        proc = subprocess.Popen("ps -e | grep flirt", encoding='utf-8', stdout=subprocess.PIPE, shell=True)

        outs = proc.communicate()[0].split('\n')
        for line in outs:
            parts = line.split()
            if len(parts) > 0:
                time = parts[2].split(':')
                secs = int(time[0])*3600 + int(time[1])*60 + int(time[2])
                if secs > 900:
                    if verb:
                        print('Killing process, flirt taking too long')
                    os.kill(int(parts[0]), signal.SIGTERM)
        
        return
    
    def get_next_avail(self, processes, verb, limit, key, s_name):
        '''
        Managues the queue for processes
        
        Parameters:
            processes : array-like
                List of processes that are being run
            verb : boolean
                Flag to turn on verbose output
            limit : int
                Number of processes that can be ran at once
            key : str
                What type of analysis is being done
            s_name : str
                Name of the script that is being run
        
        Returns:
            index : int
                Index of the in the processing queue that has become open
        '''
        
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
        '''
        Wait for the queue to empty
        
        Parameters:
            processes : array-like
                List of processes that are being run
            verb : boolean
                Flag to turn on verbose output
            key : str
                What type of analysis is being done
            s_name : str
                Name of the script that is being run
        
        Returns:
            Nothing
        '''
        
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