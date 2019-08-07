#!/usr/bin/env python

import os, sys
import pandas as pd
import numpy as np
import scipy.signal as sg
import argparse
import matplotlib.pyplot as plt
#import analysis

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

#separate patient ids and scan dates and pack into tuple
#patient ID,  (month, day, year)
patient_tuples = [(''.join(f[0:5]), (''.join(f[7:8]),''.join(f[9:10]),''.join(f[11:14]))) for f in txt_files]

print(patient_tuples)
