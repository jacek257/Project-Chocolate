'''
By: Jimi Cao, Austin Sophonsri

'''

import os
import pandas as pd
from scipy import signal as sg
from scipy import interpolate
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# instantiate the argument parser
parser = argparse.ArgumentParser()

# add positional argument
parser.add_argument("path", help="path of the folder that contains all the endtidal edit files")

# add optional arguments
parser.add_argument("-v", "--verbose", action='store_true', help="incrase output verbosity")

#get the positional arguments
args = parser.parse_args()
path = args.path

if args.verbose:
    verb = True
else:
    verb = False

# make sure the path ends with '/'
if path[-1] != '/':
    path += '/'

# all grab all the .txt files in the endtidal folder
txt_files = [file for file in os.listdir(path) if file.endswith('.txt')]

# set the size of the graphs
sns.set(rc={'figure.figsize':(20,10)})

for file in txt_files:
    if verb:
        print("Checking ", file)

    # combine file name with path to get the full path of the file
    f_path = path+file

    # make the directory that the contrast data will be saved to if directory doesn't exist
    if not os.path.exists(f_path[:-4]):
        os.mkdir(f_path[:-4])
    else:
        continue

    # read in the data from the text files, but only keep the Time, O2, and CO2 data
    df = pd.read_csv(f_path, sep='\t|,', names=['Time', 'O2', 'CO2', 'thrw', 'away'], usecols=['Time', 'O2', 'CO2'], index_col=False, engine='python')

    # drop rows with missing data
    df = df.dropna()

    if df.empty:
        print(file, "dropped. Bad data")
        os.rmdir(f_path[:-4])
        continue
    
    # need to scale CO2 data is necessary
    if df.CO2.max() < 1:
        df.CO2 = df.CO2 * 100

    # make a loop for user confirmation that O2 peak detection is good
    bad = True
    prom = 1
    while bad:
        # get the troughs of the O2 data
        low_O2, _ = sg.find_peaks(df.O2.apply(lambda x:x*-1), prominence=prom)
#        O2_interp = interpolate(df.iloc[low_O2].Time, df.iloc[low_O2].O2, fill_value='extrapolate')

        # create scatterplot of all O2 data
        if verb:
            print("Creating O2 plot ", file)
        sns.lineplot(x='Time', y='O2', data=df, linewidth=1, color='b')

        # get the data points of peak
        O2_df = df.iloc[low_O2]
        
        # linear interpolate the number of data points to match the scan Time
        O2_fxn = interpolate.interp1d(O2_df.Time, O2_df.O2, fill_value='extrapolate')
        et_O2 = O2_fxn(df.Time)
        
        # add peak overlay onto the scatterplot
        sns.lineplot(x=df.Time, y=et_O2, linewidth=2, color='g')
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
#        CO2_interp = interpolate(df.iloc[low_CO2].Time, df.iloc[high_CO2].CO2, fill_value='extrapolate')

        # create scatter of all CO2 data
        if verb:
            print('Creating CO2 plot ', file)
        sns.lineplot(x='Time', y='CO2', data=df, linewidth=1, color='b')

        # get the data points of peak
        CO2_df = df.iloc[high_CO2]
        
        # linear interpolate the number of data points to match the scan Time
        CO2_fxn = interpolate.interp1d(CO2_df.Time, CO2_df.CO2, fill_value='extrapolate')
        et_CO2 = CO2_fxn(df.Time)
        
        # add peak overlay onto the scatterplot
        sns.lineplot(x=df.Time, y=et_CO2, linewidth=2, color='r')
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

    # create subplots for png file later
    f, axes = plt.subplots(2, 1)

    # recreate the plot because plt.show clears plt
    sns.lineplot(x='Time', y='O2', data=df, linewidth=1, color='b', ax=axes[0])
    sns.lineplot(x=df.Time, y=et_O2, linewidth=2, color='g', ax=axes[0])

    # save the plot
    if verb:
        print('Saving plots for', file)
    # recreate the plot because plt.show clears plt
    sns.lineplot(x='Time', y='CO2', data=df, linewidth=1, color='b', ax=axes[1])
    sns.lineplot(x=df.Time, y=et_CO2, linewidth=2, color='r', ax=axes[1])

    save_path = save_path = path+file[:len(file)-4]+'/graph.png'
    f.savefig(save_path)
    if verb:
        print('Saving complete')
    f.clf()

    # since the find_peaks only returns the index that the peak is found, we need to grab the actual data point
    O2_df = df.iloc[low_O2].O2
    CO2_df = df.iloc[high_CO2].CO2

    # make the file name and save the O2 and CO2 data into their corresponding file
    if verb:
        print("Saving O2 data for ", file)
    save_path = path+file[:len(file)-4]+'/O2_contrast.txt'
    np.savetxt(save_path, et_O2, delimiter='\t')

    if verb:
        print('Saving CO2 data for ', file)
    save_path = path+file[:len(file)-4]+'/CO2_contrast.txt'
    np.savetxt(save_path, et_CO2, delimiter='\t')

    if verb:
        print()
