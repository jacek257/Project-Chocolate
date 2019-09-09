#!/usr/bin/env python


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.signal as sg
import scipy.interpolate as interp
from scipy.fftpack import fft, ifft
from tqdm import tqdm

class fft_analysis:
    """docstring for fft_analysis."""

    def getDataArray(self, f_path):
        df = pd.read_csv(f_path, sep='\t', names=['Time', 'O2', 'CO2', 'thrw', 'away'],
                     usecols=['Time', 'O2', 'CO2'], index_col=False)
        return df[["Time","O2","CO2"]]

    def fourier_trans(self, t_step, data):
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

    def fourier_filter(self, time_series, data, low_f, high_f, TR):
        """
        Driver module: runs fourier_trans() and my_filter() and downsamples

        time_steps = time_step list
        data = data to be analyzed
        low_f = lower frequency bound
        high_f = upper frequency bound
        TR = repetition time: found in BOLD .json
        """
        freq, power, disp = self.fourier_trans(time_series[1], data)
        pre_invert = self.my_filter(low_f,high_f, freq, power)
        inverted = ifft(pre_invert).real


        if(time_series[len(time_series)-1]<10):
             time_series = time_series*60
        else:
             time_series = time_series

        resample_ts = np.arange(0,480,TR)
        resampler = interp.interp1d(time_series, inverted, fill_value="extrapolate")
        return (resampler(resample_ts))

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
        return inverted

    def plotFourier(self, freq_dom, plottable):
        """
        self explanatory in name
        """
        plt.figure(figsize=(20,10))
        plt.semilogx(x = freq_dom, y = plottable)

class stat_utils:
    """docstring for stat_utils."""

    def showMe(*plots):
        """
        plots all data passed as a parameter
        x-axis is the integer index of each data list

        return:
            none
        """
        plt.figure(figsize=(20,10))
        for p in plots:
            plt.plot(p)
        plt.show()

    def save_meants(self, meants, peak_prediction, f_path):
        #generate predicted plot and meants plot
        plt.plot(np.linspace(0,480, len(meants)), meants, label='meants')
        plt.plot(np.linspace(0,480, len(meants)), peak_prediction, label='predicted')
        plt.legend()
        plt.savefig(f_path[:-4]+'/regression_plot.png')

    def save_plots(self, df, O2, O2_shift, CO2, CO2_shift, meants, f_path, key, verb, TR):
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

        #construct interpolation time_step series
        resample_ts = np.arange(0,480,TR)

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
            print('Creating plots to be saved')
        # create subplots for png file later
        f, axes = plt.subplots(2, 3)
        
        if df.Time.max() < 10:
            df.Time = df.Time * 60

        sns.lineplot(x='Time', y='O2', data=df, linewidth=1, color='b', ax=axes[0, 0])
        sns.lineplot(x=resample_ts, y=O2, linewidth=2, color='g', ax=axes[0, 0])

        sns.lineplot(x='Time', y='CO2', data=df, linewidth=1, color='b', ax=axes[1, 0])
        sns.lineplot(x=resample_ts, y=CO2, linewidth=2, color='r', ax=axes[1, 0])

        sns.lineplot(x=resample_ts, y=meants_norm, color='b', ax=axes[0,1])
        sns.lineplot(x=resample_ts, y=O2_norm, color='g', ax=axes[0,1])

        sns.lineplot(x=resample_ts, y=meants_norm, color='b', ax=axes[1,1])
        sns.lineplot(x=resample_ts, y=CO2_norm, color='r', ax=axes[1,1])

        sns.lineplot(x=resample_ts, y=meants_norm, color='b', ax=axes[0,2])
        sns.lineplot(x=resample_ts, y=O2_shift_norm, color='g', ax=axes[0,2])

        sns.lineplot(x=resample_ts, y=meants_norm, color='b', ax=axes[1,2])
        sns.lineplot(x=resample_ts, y=CO2_shift_norm, color='r', ax=axes[1,2])

        # save the plot
        if verb:
            print('Saving plots for', f_path)

        save_path = save_path = f_path[:-4]+'/' + key + 'graph.png'
        f.savefig(save_path)
        if verb:
            print('Saving complete')
#        plt.show()
        f.clf()
        plt.close(fig=f)

        if verb:
            print()

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

    def peak_four(self, df, verb, file, TR, trough=False):
        
        # set the size of the graphs
        sns.set(rc={'figure.figsize':(20,10)})
        
        f_O2 = fft_analysis().fourier_filter_no_resample(df.Time, df.O2, 3, 35)
        f_CO2 = fft_analysis().fourier_filter_no_resample(df.Time, df.CO2, 3, 35)          
        
        if df.Time.max() < 10:
            df.Time = df.Time * 60

        # get the troughs of the O2 data
        O2_data, _ = sg.find_peaks(df.O2.apply(lambda x:x*-1), prominence=2)
        CO2_df = df.drop(columns=['CO2'])
        O2_df = df.iloc[O2_data]
        
        f_O2_resamp = interp.interp1d(df.Time, f_O2, fill_value="extrapolate")
        f_O2_final = f_O2_resamp(O2_df.Time)
        
        O2_df['cmp'] = O2_df.O2 < f_O2_final
        
        O2_df = O2_df[O2_df.cmp == True].drop(columns=['cmp'])
        
        if trough:
            CO2_data, _ = sg.find_peaks(df.CO2.apply(lambda x:x*-1), prominence=3)
        else:
            CO2_data, _ = sg.find_peaks(df.CO2, prominence=3) 
        
        CO2_df = df.drop(columns=['O2'])
        CO2_df = CO2_df.iloc[CO2_data]
        
        f_CO2_resamp = interp.interp1d(df.Time, f_CO2, fill_value="extrapolate")
        f_CO2_final = f_CO2_resamp(CO2_df.Time)
        
        if trough:
            CO2_df['cmp'] = CO2_df.CO2 < f_CO2_final
        else:
            CO2_df['cmp'] = CO2_df.CO2 > f_CO2_final
        
        CO2_df = CO2_df[CO2_df.cmp == True].drop(columns=['cmp'])

        CO2_resamp = interp.interp1d(CO2_df.Time, CO2_df.CO2, fill_value='extrapolate')
        CO2_final = CO2_resamp(np.arange(0,480,TR))
        O2_resamp = interp.interp1d(O2_df.Time, O2_df.O2, fill_value='extrapolate')
        O2_final = O2_resamp(np.arange(0,480,TR))

        return CO2_final, O2_final

    def get_peaks(self, df, length, verb, file, TR, trough=False):
        """
        Get the peaks and troughs of CO2 and O2

        Parameters:
            df: dataframe
                data structure that holds the data
            length: int
                lenght of the BOLD data that the et_CO2 and et_O2 will be resampled to
            verb: boolean
                flag for verbose output
            file: string
                file name to be displayed during verbose output
            TR: float
                repetition time, found in BOLD.json
            trough: boolean
                get the troughs of CO2 data

        Returns:
            et_O2: array-like
                the end-tidal O2 data
            et_CO2: array-like
                the end-tidal CO2 data
        """

        # set the size of the graphs
        sns.set(rc={'figure.figsize':(20,10)})

        # make a loop for user confirmation that O2 peak detection is good
        bad = True
        prom = 2
        while bad:
            # get the troughs of the O2 data
            O2_data, _ = sg.find_peaks(df.O2.apply(lambda x:x*-1), prominence=prom)

            # create scatterplot of all O2 data
            if verb:
                print("Creating O2 plot ", file)
            sns.lineplot(x='Time', y='O2', data=df, linewidth=1, color='b')

            # get the data points of peak
            O2_df = df.iloc[O2_data]

            # add peak overlay onto the scatterplot
            sns.lineplot(x='Time', y='O2', data=O2_df, linewidth=2, color='g')
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
            # get the troughs of the O2 data
            if trough:
                CO2_data, _ = sg.find_peaks(df.CO2.apply(lambda x:x*-1), prominence=prom)
            else:
                CO2_data, _ = sg.find_peaks(df.CO2, prominence=prom)

            # create scatter of all CO2 data
            if verb:
                print('Creating CO2 plot ', file)
            sns.lineplot(x='Time', y='CO2', data=df, linewidth=1, color='b')

            # get the data points of peak
            CO2_df = df.iloc[CO2_data]

            # add peak overlay onto the scatterplot
            sns.lineplot(x='Time', y='CO2', data=CO2_df, linewidth=2, color='r')
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

        CO2_resamp = interp.interp1d(CO2_df.Time, CO2_df.CO2, fill_value='extrapolate')
        CO2_final = CO2_resamp(np.linspace(0, 480, length))
        O2_resamp = interp.interp1d(O2_df.Time, O2_df.O2, fill_value='extrapolate')
        O2_final = O2_resamp(np.linspace(0, 480, length))

        return CO2_final, O2_final

    def get_wlen(self, sig_time, sig):

        freq,_,power = fft_analysis().fourier_trans(sig_time[1], sig)
        window_it = np.argmax(power[25:500])+25
        freq_val = freq[window_it] # this is the most prominent frequency
        window_mag = 1/freq_val

        window_length = 0
        for i, t in enumerate(sig_time):
            if t > window_mag:
                window_length = i-1
                break

        return window_length


    def block_signal(self, sig_time, sig, TR):
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
        
        if sig_time.max() < 10:
            sig_time = sig_time * 60

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

        # return filtered
        signal_resampler = interp.interp1d(sig_time, filtered, fill_value='extrapolate')
        signal_resampled = signal_resampler(np.arange(0,480,TR))

        return signal_resampled

class shifter:
    """
    docstring for shifter.
    acknowledgements:
    https://ws680.nist.gov/publication/get_pdf.cfm?pub_id=901379 (centroid analysis)

    Prefer to use cross-correlaton over centroid
    """

    def get_centroid(self, t_series, data_series, window_coeff, poly_order):
        """
        centroid is a signal weighted average of time. We are essentially calculating the temporal middle of the signal

        params:
            t_series (iterable) = time points of data_series
            data_series (iterable) = the signal intensity series
            window_coeff (int) = filter window size
            poly_order (int) = the order of savgol fit polynomial

        return:
            (float)
        """
        #set buffer to 0
        buffer = 0.0
        #pack t_series and d_series together and calculate square weighted sum
        for t, d in zip(t_series, sg.savgol_filter(data_series, window_coeff, poly_order)):
            buffer += (d**2) * t
        #return square weighted average
        return buffer/(np.sum(data_series**2))

    def align_centroid(self, base_time, base_sig, time1, sig1, time2, sig2):
        """
        aligns several (3) time series based on their relative temporal centroids

        params:
            base_time = BOLD
            rest = self explanatory
            *time* (iterable) = time points / sampling points
            *sig* (iterable) = sampling intensity at each time/sampling points

        warnings:
            [*]sig[*] must have equal length to [*]time[*]

        returns:
            (dictionary) {base_time, base_sig, sig1, sig2}
        """
        #get all centroids
        base_centroid = self.get_centroid(base_time, base_sig)
        sig1_centroid = self.get_centroid(time1, sig1)
        sig2_centroid = self.get_centroid(time2, sig2)

        #shift times and construct interpolators
        sig1_resampler = interp.interp1d(time1 - sig1_centroid + base_centroid, sig1)
        sig2_resampler = interp.interp1d(time2 - sig2_centroid + base_centroid, sig2)

        #interpolate signals
        sig1_exact_aligned = sig1_resampler(base_time)
        sig2_exact_aligned = sig2_resampler(base_time)

        return {'base_time' : base_time, 'base_sig' : base_sig, 'sig1' : sig1_exact_aligned, 'sig2' : sig2_exact_aligned}

    def get_cross_correlation(self, base, sig):
        """
        Params:
            base (iterable) : reference signal (used to align signal)
            sig (iterable) : signal that needs alignment

        Returns:
            (float) : shift value of signal
        """
        limit = int(10 * len(base)/480)
        correlation_series = sg.correlate(base, sig)
        correlation_series = correlation_series[len(correlation_series)//2 - limit:len(correlation_series)//2 + limit]

        shift_index = np.argmax(correlation_series) - (len(correlation_series)//2)
        shift_value = 480/len(base) * shift_index
        return shift_value

    def corr_align(self, base, other):
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
        base_norm = sg.savgol_filter(base, 5, 3)
        base_norm -= base_norm.mean()
        base_norm /= base_norm.std()

        other_norm = other - other.mean()
        other_norm /= other_norm.std()

        #get shifts
        shift = self.get_cross_correlation(base_norm, other_norm)
        #construct resampler
        resamp = interp.interp1d(np.linspace(0,480, len(base_norm))+shift, other, fill_value='extrapolate')
        shifted = resamp(np.linspace(0, 480, len(base_norm)))

        return shifted

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

    def stochastic_optimize_GLM(self, S1, S2, B, init_tuple = (0,0,0), descent_speed=.1, lifespan = 100, p_factor = .9):
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
