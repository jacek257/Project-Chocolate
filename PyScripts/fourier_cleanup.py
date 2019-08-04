'''
By: Jimi Cao, Austin Sophonsri

'''

import os
import pandas as pd
import numpy as np
from scipy import signal as sg
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import analysis


# instantiate the argument parser
parser = argparse.ArgumentParser()

# add positional argument
parser.add_argument("path", help="path of the folder that contains all the endtidal edit files")

# add optional arguments
parser.add_argument("-v", "--verbose", action='store_true', help="incrase output verbosity")
parser.add_argument("-g", "--graph", action='store_true', help='display graphs as it is generated')

#get the positional arguments
args = parser.parse_args()
path = args.path

if args.verbose:
    verb = True
else:
    verb = False

if args.graph:
    graph = True
else:
    graph = False

# make sure the path ends with '/'
if path[-1] != '/':
    path += '/'

# all grab all the .txt files in the endtidal folder
txt_files = [file for file in os.listdir(path) if file.endswith('.txt')]

# set the size of the graphs
sns.set(rc={'figure.figsize':(20,10)})

for file in txt_files:
    if verb:
        print("Processing ", file)

    # combine file name with path to get the full path of the file
    f_path = path+file
    # read in the data from the text files, but only keep the Time, O2, and CO2 data
    df = pd.read_csv(f_path, sep='\t|,', names=['Time', 'O2', 'CO2', 'thrw', 'away'], usecols=['Time', 'O2', 'CO2'], index_col=False, engine='python')

    df = df.dropna()

    if df.empty:
        print(file, " dropped. File did not contain the correct amount of data")
        continue

    # make the directory that the contrast data will be saved to if directory doesn't exist
    if not os.path.exists(f_path[:-4]):
        os.mkdir(f_path[:-4])


    # fourier transform and filter and invert O2
    low_O2 = analysis.fourier_filter(df.Time[1], df.O2,3,25)
    fft_O2 = pd.DataFrame({'Time': df.Time,'O2' : low_O2})

    # create scatterplot of all O2 data
    if verb:
        print("Creating O2 plot ", file)
    sns.lineplot(x='Time', y='O2', data=df, linewidth=1, color='b')

    # add peak overlay onto the scatterplot
    sns.lineplot(x='Time', y='O2', data=fft_O2, linewidth=4, color='g')

    # save the plot
    if verb:
        print("Saving O2 plot ", file)
    save_path = path+file[:len(file)-4]+'/O2_graph.png'
    plt.savefig(save_path)
    if graph:
        plt.show()
    plt.close()

    # fourier transform and filter and invert CO2
    high_CO2 = analysis.fourier_filter(df.Time[1], df.CO2,3,25)
    fft_CO2 = pd.DataFrame({'Time': df.Time,'CO2' : high_CO2})


    # create scatter of all CO2 data
    if verb:
        print('Creating CO2 plot ', file)
    sns.lineplot(x='Time', y='CO2', data=df, linewidth=1, color='b')

    # add peak overlay onto the scatterplot
    sns.lineplot(x='Time', y='CO2', data=fft_CO2, linewidth=4, color='r')

    # save the plot
    if verb:
        print('Saving CO2 plot ', file)
    save_path = save_path = path+file[:len(file)-4]+'/CO2_graph.png'
    plt.savefig(save_path)
    if graph:
        plt.show()
    plt.close()

    # since the find_peaks only returns the index that the peak is found, we need to grab the actual data point
    # O2_df = df.iloc[low_O2].O2
    # CO2_df = df.iloc[high_CO2].CO2

    # make the file name and save the O2 and CO2 data into their corresponding file
    if verb:
        print("Saving O2 data for ", file)
    save_path = path+file[:len(file)-4]+'/O2_contrast.txt'
    # O2_df.to_csv(path_or_buf=save_path, sep='\t', header=False, index=False)
    np.savetxt(save_path, low_O2, delimiter='\n')

    if verb:
        print('Saving CO2 data for ', file)
    save_path = path+file[:len(file)-4]+'/CO2_contrast.txt'
    #CO2_df.to_csv(path_or_buf=save_path, sep='\t', header=False, index=False)
    np.savetxt(save_path, high_CO2, delimiter='\n')
    
    if verb:
        print()
