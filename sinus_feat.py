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
import pandas as pd
import alter_nii as alter

import parallel_processing as pp

# Limit the number of number of processes that can be run at once, do not want to consume all CPU processing power
cores = multiprocessing.cpu_count()
limit = cores - 5 if cores > 8 else 1
processes = [None] * limit

# instantiate the argument parser
parser = argparse.ArgumentParser()

# add optional arguments
parser.add_argument("-v", "--verbose", action='store_true', help="incrase output verbosity")
parser.add_argument("-g", "--graph", action='store_true', help="display all intermediary graphs")
parser.add_argument("-o", "--overwrite", action='store_true', help='overwrite existing processed gas data')

#get the positional arguments
args = parser.parse_args()

verb = True if args.verbose else False
graph = True if args.graph else False
over = True if args.overwrite else False

######################################
# TODO: Change paths after database organization
root = '/media/ke/8tb_part2/FSL_work/all_info'
melodic = '~/Desktop/melodic_cmd'

# Create dataframe to store information
df = pd.DataFrame(columns=['pt', 'BOLD', 'zstat', 'signal', 'corrected BOLD'])

# Get all relevant BOLD folders
pts = []
bold_files = []
for folder in os.listdir(root):
    if int(folder.split('_')[-1]) > 20200000 and folder.split('_')[0][:2] == 'WH':
        pts.append(folder)
        for file in os.listdir(root+'/'+folder+'/BOLD'):
            if file.endswith('.nii'):
                bold_files.append(root+'/'+folder+'/BOLD/'+file)
    elif folder.split('_')[0][:3] == 'CVR':
        pts.append(folder)
        for file in os.listdir(root+'/'+folder+'/BOLD'):
            if file.endswith('.nii'):
                bold_files.append(root+'/'+folder+'/BOLD/'+file)

df['pt'] = pts
df['BOLD'] = bold_files

# Pass the BOLD images through melodic
for i in range(len(df)):
    if verb:
        print('Running melodic on', df['pt'][i])
    output_path = '/home/ke/Desktop/melodic_cmd/'+df['pt'][i]
    if os.path.exists(output_path):
        print('\tFolder already exists')
        if over:
            if verb:
                print('\t\tOverwritting')
                subprocess.run(['rm', '-rf', output_path])
        else:
            continue
    index = pp.get_next_avail(processes, verb, 'melodic', limit)
    processes[index] = subprocess.Popen(['melodic', '-i', df['BOLD'][i], '-o', output_path , '--Oall'])
    df['zstat'][i] = output_path
    
# construct the processed nifti directory if necessary
for i in range(len(df)):
    processed_dir = root + '/' + df['pt'][i] +'/BOLD_processed/'
    if(not os.path.exists(processed_dir)):
        os.mkdir(processed_dir)
    
    # check if corrected BOLD has already been made, if not make it
    if verb:
        print('\tCreating files if necessary for', df['pt'][i])
    corrected_BOLD_path = processed_dir+df['BOLD'][i][df['BOLD'][i].rfind('/')+1:-4]+'_timed_demotioned.nii.gz'
    if os.path.exists(corrected_BOLD_path):
        df['corrected BOLD'][i] = processed_dir+df['BOLD'][i][df['BOLD'][i].rfind('/')+1:-4]+'_timed_demotioned.nii.gz'
    else:
        time_correct = processed_dir+df['BOLD'][i][df['BOLD'][i].rfind('/')+1:-4]+'_timed.nii'
        if(not os.path.exists(time_correct+'.gz')): # must add .gz because slicetimer adds it automatically
            if verb:
                print('\t\tNo slice timing correction. Creating timing correction')
            subprocess.run(['slicetimer', '-i', df['BOLD'][i], '-o', time_correct, '--odd'])
        
        motion_correct = time_correct[:-4]+'_demotioned.nii'
        if(not os.path.exists(motion_correct+'.gz')):
            if verb:
                print('\t\tNo motion correction. Creating motion correction')
            subprocess.run(['mcflirt', '-in', time_correct, '-out', motion_correct])
        
        df['corrected BOLD'][i] = motion_correct+'.gz'
    
pp.wait_remaining(processes, verb, 'melodic')

# Main Body
for i in range(len(df)):
    if verb:
        print(df['corrected BOLD'][i])
    nii = nib.load(df['corrected BOLD'][i])
    nii_image = nii.dataobj
    
    # remove the first 3 slices to allow for intensity calibration
    nii_image = nii_image[:, :, :, 3:]
    
    # condense the 4D data into 3D data
    nii_max = alter.condense(nii_image, 'max')    
    # create brain mask based off the max intensities
    max_mask = alter.create_mask(nii_max, 0.05, 0.95, opening=False, verb=verb)
    # extract brain
    max_brain = nii_max * max_mask
    
    if graph:
        # display graphs
        alter.display_brain_by_slice(max_brain, max_brain, df['pt'][i], 'max')
    
    # create a mask based off the median intensities
    nii_med = alter.condense(nii_image, 'median')
    med_mask = alter.create_mask(nii_med, 0.05, 0.95, opening=False, verb=verb)
    # extract brain
    med_brain = nii_med * med_mask
    if graph:
        # display graphs
        alter.display_brain_by_slice(med_brain, med_mask, df['pt'][i], 'med')
    
    # create a mask based off the min intensities
    nii_min = alter.condense(nii_image, 'min')
    min_mask = alter.create_mask(nii_min, 0.05, 0.95, opening=False, verb=verb)
    # extract_brain
    min_brain = nii_min * min_mask
    # display graphs
    if graph:
        alter.display_brain_by_slice(min_brain, min_mask, df['pt'][i], 'min')
        
    final_mask = max_mask.astype(int)+med_mask.astype(int)+min_mask.astype(int)
    final_mask[final_mask <= 1] = 0
    final_mask[final_mask > 1] = 1
    
    range_nii = alter.condense(nii_image, 'range')
    final_brain = range_nii*final_mask
    if graph:
        alter.display_brain_by_slice(final_brain, final_mask, df['pt'][i], 'range')
    
    sinus_mask = alter.create_mask(final_brain, m=0.90, M=0.98, opening=False, exclude_zeros=False)
    sinus = final_brain*sinus_mask    
    
    sinus_sig = []
    for i in range(nii.shape[-1]):
        sinus_sig[i] = final_brain[:,:,:,i].mean()
    if graph:
        plt.plot(sinus_sig)
        plt.show()
