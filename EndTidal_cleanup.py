'''
By: Jimi Cao, Austin Sophonsri

'''

import os
import pandas as pd
from scipy import signal as sg
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

    df = df.dropna()

    if df.empty or df.CO2.max() < 1:
        print(file, "dropped. Bad data")
        continue

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

        # add peak overlay onto the scatterplot
        sns.lineplot(x='Time', y='O2', data=df.iloc[low_O2], linewidth=2, color='g')
        plt.show()
        plt.close()

        # ask user if the peak finding was good
        ans = input("Was the output good enough (y/n)? \nNote: anything not starting with 'y' is considered 'n'.\n")
        bad = True if ans == '' or ans[0].lower() != 'y' else False
        if bad:
            print("The following variables can be changed: ")
            print("    1. prominence - Required prominence of peaks. Type: int")
            print("    2. width      - Required distance between peaks. Type: int")
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

        # add peak overlay onto the scatterplot
        sns.lineplot(x='Time', y='CO2', data=df.iloc[high_CO2], linewidth=2, color='r')
        plt.show()
        plt.close()

        # ask user if the peak finding was good
        ans = input("Was the output good enough (y/n)? \nNote: anything not starting with 'y' is considered 'n'.\n")
        bad = True if ans == '' or ans[0].lower() != 'y' else False
        if bad:
            print("The following variables can be changed: ")
            print("    1. prominence - Required prominence of peaks. Type: int")
            print("    2. width      - Required distance between peaks. Type: int")
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
    sns.lineplot(x='Time', y='O2', data=df.iloc[low_O2], linewidth=2, color='g', ax=axes[0])

    # save the plot
    if verb:
        print('Saving plots for', file)
    # recreate the plot because plt.show clears plt
    sns.lineplot(x='Time', y='CO2', data=df, linewidth=1, color='b', ax=axes[1])
    sns.lineplot(x='Time', y='CO2', data=df.iloc[high_CO2], linewidth=2, color='r', ax=axes[1])

    save_path = save_path = path+file[:len(file)-4]+'/graph.png'
#    plt.savefig(save_path)
    f.savefig(save_path)
    if verb:
        print('Saving complete')
    f.clf()
#    plt.close()

    # since the find_peaks only returns the index that the peak is found, we need to grab the actual data point
    O2_df = df.iloc[low_O2].O2
    CO2_df = df.iloc[high_CO2].CO2

    O2_resamp = sg.resample(O2_df, 320)
    CO2_resamp = sg.resample(CO2_df, 320)

    # make the file name and save the O2 and CO2 data into their corresponding file
    if verb:
        print("Saving O2 data for ", file)
    save_path = path+file[:len(file)-4]+'/O2_contrast.txt'
    np.savetxt(save_path, O2_resamp, delimiter='\t')

    if verb:
        print('Saving CO2 data for ', file)
    save_path = path+file[:len(file)-4]+'/CO2_contrast.txt'
    np.savetxt(save_path, CO2_resamp, delimiter='\t')

    if verb:
        print()
