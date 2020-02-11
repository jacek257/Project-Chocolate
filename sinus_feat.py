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

cores = multiprocessing.cpu_count()
limit = cores - 5 if cores > 8 else 1
processes = [None] * limit

root = '/media/ke/8tb_part2/FSL_work/all_info'

bold_folders = []

for folder in os.listdir(root):
    if int(folder.split('_')[-1]) > 20200000 and folder.split('_')[0][:2] == 'WH':
        bold_folders.append(folder)
    elif folder.split('_')[0][:3] == 'CVR':
        bold_folders.append(folder)

bold_files = []

for folder in bold_folders:
    for file in os.listdir(root+'/'+folder+'/BOLD'):
        if file.endswith('.nii'):
            bold_files.append(root+'/'+folder+'/BOLD/'+file)

for file in bold_files:
    pt_info = file.split('/')
    pt = pt_info[-3]
    print(pt)