'''
By: Austin Sophonsri

'''

import os
import argparse
import pandas as pd

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

verb = (True if args.verbose else False)
graph = (True if args.graph else  False)

# make sure the path ends with '/'
if path[-1] != '/':
    path += '/'

# all grab all the .txt files in the endtidal folder
txt_files = [file for file in os.listdir(path) if file.endswith('.txt')]

#separate patient ID and scan date and pack into tuple
p_df = pd.DataFrame({
                'Cohort':[''.join(f[0:2]) for f in txt_files],
                'ID':[''.join(f[3:6]) for f in txt_files],
                'Month':[''.join(f[7:9]) for f in txt_files],
                'Day':[''.join(f[9:11]) for f in txt_files],
                'Year':[''.join(f[11:15]) for f in txt_files]
             })

print(p_df.head())
#create patient bold scan listdir (is a list not DataFrame)
patient_BOLDS_header = [p_df.Cohort[i]+p_df.ID[i]+'_BOLD_'+p_df.Year[i]+p_df.Month[i]+p_df.Day[i]
                    for i in range(len(p_df))]

print(patient_BOLDS_header[:5])

#kinda the same thing as below
# BOLD_files = [file for it, file in enumerate(os.listdir(BOLD_path)) if (file.endswith('.nii') and file.startswith(patient_BOLDS_header[it]))]

#but construct paths from patient dataframe
#so a multi line for loop may be necessary

#Bold_files =














#
