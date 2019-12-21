#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:13:00 2019

@author: Jimi Cao
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.interpolate as interp
from sklearn.linear_model import Ridge
from scipy import stats

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
    