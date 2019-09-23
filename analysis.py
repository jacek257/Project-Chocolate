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

class fft_analysis:
    """docstring for fft_analysis."""

    def fourier_trans(self, spacing, data):
        """
        returns a tuple: (frequency domain, Power spectra, abs(power_spectra))

        spacing = the distance between data points
        data = series to be analyzed
        """

        N = len(data)

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
        for i,f in enumerate(freq_dom):
            if (f >= f_low and f<= f_high):
                cp[i] = 0
                cp[-i] = 0
        return np.copy(cp)

    def fourier_filter(self, time_series, data, low_f, high_f, tr, time_points):
        """
        Driver module: runs fourier_trans() and my_filter() and downsamples

        time_steps = time_step list
        data = data to be analyzed
        low_f = lower frequency bound
        high_f = upper frequency bound
        TR = repetition time: found in BOLD .json
        """
#
#        if(time_series.max()<10):
#             time_series = time_series*60
#        else:
#             time_series = time_series
        
        freq, power, disp = self.fourier_trans(max(time_series)/len(time_series), data)
        pre_invert = self.my_filter(low_f,high_f, freq, power)
        inverted = ifft(pre_invert).real

#        resample_ts = time_points
#        resampler = interp.interp1d(time_series, inverted, fill_value="extrapolate")
        
        return pd.DataFrame({'Time' : time_series,
                             'Data' : inverted})
#                             'Data' : resampler(time_series)})


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

    def save_plots(self, df, O2_time, O2, O2_shift, O2_correlation, CO2_time, CO2, CO2_shift, CO2_correlation, meants, f_path, key, verb, time_points, TR):
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

        # normalize data
        meants_norm = meants - meants.mean()
        meants_norm /= meants_norm.std()

        CO2_norm = CO2 - CO2.mean()
        CO2_norm /= CO2.std()

        CO2_shift_norm = CO2_shift - CO2_shift.mean()
        CO2_shift_norm /= CO2_shift_norm.std()

        O2_norm = O2 - O2.mean()
        O2_norm /= O2.std()

        O2_shift_norm = O2_shift - O2_shift.mean()
        O2_shift_norm /= O2_shift_norm.std()


        if verb:
            print('Creating O2 plots to be saved')
        # create subplots for png file later
        f, axes = plt.subplots(2, 2)
        
#        if df.Time.max() < 10:
#            df.Time = df.Time * 60

        sns.lineplot(x='Time', y='O2', data=df, linewidth=1, color='b', ax=axes[0, 0])
        sns.lineplot(x=O2_time, y=O2, linewidth=2, color='g', ax=axes[0, 0])

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0,1])
        sns.lineplot(x=O2_time, y=O2_norm, color='g', ax=axes[0,1])
        
        sns.lineplot(x=np.arange(-len(O2_correlation)//2, len(O2_correlation)//2)+1, y=O2_correlation, ax=axes[1,0])

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1,1])
        sns.lineplot(x=time_points, y=O2_shift_norm, color='g', ax=axes[1,1])

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

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[0,1])
        sns.lineplot(x=CO2_time, y=CO2_norm, color='r', ax=axes[0,1])
        
        sns.lineplot(x=np.arange(-len(CO2_correlation)//2, len(CO2_correlation)//2)+1, y=CO2_correlation, ax=axes[1,0])

        sns.lineplot(x=time_points, y=meants_norm, color='b', ax=axes[1,1])
        sns.lineplot(x=time_points, y=CO2_shift_norm, color='r', ax=axes[1,1])

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

    def get_r2(self, sig_fit, sig_obs):
        """
        gets r^2 (coefficient of determination) of sig_fit w.r.t. sig_obs

        inputs:
            sig_fit (iterable) = fitted/predicted signal
            sig_obs (iterable) = observed signal / raw data

        warnings:
            sig's must have the same lengths

        return:
            (float)
        """
        if(len(sig_fit) != len(sig_fit)):
            print("Signals have different lengths: ", len(sig_fit) , ' &', len(sig_obs))
        else:
            sig_obs_var = np.mean(sig_obs)
            return 1-np.sum((sig_fit-sig_obs)**2)/np.sum((sig_obs-sig_obs_var)**2)



class peak_analysis:
    """docstring for peak_analysis."""

    def peak_four(self, df, verb, file, tr, time_pts, trough):
        
        # set the size of the graphs
        sns.set(rc={'figure.figsize':(20,10)})
#        
#        f_O2 = fft_analysis().fourier_filter_no_resample(df.Time, df.O2, 3, 35)
#        f_CO2 = fft_analysis().fourier_filter_no_resample(df.Time, df.CO2, 3, 35)         
        
        f_O2 = fft_analysis().fourier_filter(df.Time, df.O2, 2/60, 25/60, tr, time_pts)
        f_CO2 = fft_analysis().fourier_filter(df.Time, df.CO2, 2/60, 25/60, tr, time_pts)
        
#        if df.Time.max() < 10:
#            df.Time = df.Time * 60

        # get the troughs of the O2 data
        O2_data, _ = sg.find_peaks(df.O2.apply(lambda x:x*-1), prominence=2)
        O2_df = df.drop(columns=['CO2'])
        O2_df = O2_df.iloc[O2_data]
        
        f_O2_resamp = interp.interp1d(f_O2.Time, f_O2.Data, fill_value='extrapolate')
        f_O2_final = f_O2_resamp(O2_df.Time)
        
        O2_df['cmp'] = O2_df.O2 < f_O2_final
        
        O2_valid_df = O2_df[O2_df.cmp == True]
        O2_resamp = interp.interp1d(O2_valid_df.Time, O2_valid_df.O2, fill_value='extrapolate')
        O2_final_df = pd.DataFrame({'Time' : df.Time,
                                    'Data' : O2_resamp(df.Time)})
        
        if trough:
            CO2_data, _ = sg.find_peaks(df.CO2.apply(lambda x:x*-1), prominence=4)
        else:
            CO2_data, _ = sg.find_peaks(df.CO2, prominence=3) 
        
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
        CO2_final_df = pd.DataFrame({'Time' : df.Time,
                                     'Data' : CO2_resamp(df.Time)})

#        CO2_resamp = interp.interp1d(CO2_df.Time, CO2_df.CO2, fill_value='extrapolate')
#        CO2_final = CO2_resamp(time_pts)
#        O2_resamp = interp.interp1d(O2_df.Time, O2_df.O2, fill_value='extrapolate')
#        O2_final = O2_resamp(time_pts)
#
#        return CO2_final, O2_final
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

    def get_cross_correlation(self, base, sig, scan_time):
        """
        Params:
            base (iterable) : reference signal (used to align signal)
            sig (iterable) : signal that needs alignment

        Returns:
            (float) : shift value of signal
        """
        limit = int(10 * len(base)/scan_time)
        correlation_series = sg.correlate(base, sig)
        correlation_series = correlation_series[len(correlation_series)//2-limit : len(correlation_series)//2+limit+1]
        
        shift_index = np.argmax(correlation_series) - (len(correlation_series)//2)
        shift_value = scan_time/len(base) * shift_index
        return shift_value, shift_index, correlation_series

    def corr_align(self, base, other_time, other_sig, scan_time, time_points):
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
        base_norm = base - base.mean()
        base_norm /= base_norm.std()

        other_norm = other_sig - other_sig.mean()
        other_norm /= other_norm.std()

        #get shifts
        shift, start, corr = self.get_cross_correlation(base_norm, other_norm, scan_time)
        #construct resampler
        resamp = interp.interp1d(other_time+shift, other_sig, fill_value='extrapolate')
        shifted = resamp(time_points)
        if shift > 0:
            shifted[:start] = other_sig[0]

        shifted = sg.savgol_filter(shifted, 11, 3)
        
        return shifted, corr

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