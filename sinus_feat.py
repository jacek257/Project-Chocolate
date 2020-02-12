#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:16:20 2020

@author: Jimi Cao
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import multiprocessing
import subprocess
import argparse

import parallel_processing as pp

# Limit the number of number of processes that can be run at once, do not want to consume all CPU processing power
cores = multiprocessing.cpu_count()
limit = cores - 5 if cores > 8 else 1
processes = [None] * limit

# instantiate the argument parser
parser = argparse.ArgumentParser()

# add optional arguments
parser.add_argument("-v", "--verbose", action='store_true', help="incrase output verbosity")

#get the positional arguments
args = parser.parse_args()

verb = True if args.verbose else False

######################################
# TODO: Change paths after database organization
root = '/media/ke/8tb_part2/FSL_work/all_info'
melodic = '~/Desktop/melodic_cmd'

# Get all relevant BOLD folders
bold_folders = []
for folder in os.listdir(root):
    if int(folder.split('_')[-1]) > 20200000 and folder.split('_')[0][:2] == 'WH':
        bold_folders.append(folder)
    elif folder.split('_')[0][:3] == 'CVR':
        bold_folders.append(folder)

# Get the BOLD images from the BOLD folders
bold_files = []
for folder in bold_folders:
    for file in os.listdir(root+'/'+folder+'/BOLD'):
        if file.endswith('.nii'):
            bold_files.append(root+'/'+folder+'/BOLD/'+file)

# Pass the BOld images through melodic
for file in bold_files:
    pt_info = file.split('/')
    pt = pt_info[-3]
    print(pt)
    index = pp.get_next_avail(processes, verb, 'melodic', limit)
    processes[index] = subprocess.Popen(['melodic', '-i', file, '-o', '~/Desktop/melodic_cmd/'+pt, '--Oall'])

# TODO: store the paths to thresh_zstat1.nii.gz


# TODO: take nii images and convert into np arrays

