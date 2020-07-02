#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:08:22 2020

@author: Jimi Cao
"""
import numpy as np
from nipy import labs
import cv2 as cv
import matplotlib.pyplot as plt

def condense(brain_4D, fxn='max', per=75):
    '''
    Condenses a 4D time series brain scan into a 3D volumetirc using an agregate function
    
    Parameters:
        brain_4D: numpy array
            The numpy representaion of the 4D data
        fxn: str{'max', 'min', 'med', 'avg', 'range'}, optional
            A string indicating how the condense is done
                max
                    Takes the maximum value for each voxel in the time axis
                min
                    Takes the minum value for each voxel in the time axis
                med
                    Takes the median value for each voxel in the time axis
                avg
                    Takes the average value for each voxel in the time axis
                range
                    Takes the range for each voxel in the time axis
                percentile
                    Takes the nth percentile for each voxel in the time axis
        per: int
            Only used when fxn=percentile. Specifies which percentile to take
    Returns:
        brain_3D: numpy array
            The condensed 3D data
    '''
    
    if fxn == 'max':
        brain_3D = np.percentile(brain_4D, 100, axis=-1)
    elif fxn == 'min':
        brain_3D = np.percentile(brain_4D, 0, axis=-1)
    elif fxn == 'med' or fxn =='median':
        brain_3D = np.percentile(brain_4D, 50, axis=-1)
    elif fxn == 'avg' or fxn == 'mean':
        brain_3D = np.mean(brain_4D, axis=-1)
    elif fxn == 'range':
        brain_3D = np.max(brain_4D, axis=-1) - np.min(brain_4D, axis=-1)
    elif fxn == 'percentile':
        brain_3D = np.percentile(brain_4D, per, axis=-1)
    else:
        raise NameError(fxn, 'is not one of the options for fxn. Please read doc string')
    
    return brain_3D

def create_mask(brain, lower, upper, opening, verb, ex_zeros=True):
    '''
    Create a mask to extract only the brain from the scan i.e. skull strip, remove eyeballs etc.
    
    brain: numpy array
        Brain image to be masked
    lower: float
        lower percentage of intensity to be thrown out >= 0.05
    upper: float
        upper percentage of intensity to be thrown out <= 0.95
    opening: boolean
        to indicate if the mask should have openings in it or not
    verbose: boolean
        print debuging lines
    
    Returns:
        mask: numpy array
            The mask of the extracted brain
    '''
    mask_init = labs.mask.compute_mask(brain, m=lower, M=upper, opening=opening, exclude_zeros=ex_zeros)
    mask = __grow_mask(mask_init, verb)
    return mask
    
def __grow_mask(mask, verb):
    temp_mask = np.zeros_like(mask)
    for z in range(mask.shape[-1]):
        mask_part = np.zeros_like(mask[:,:,z])
        labels, stats = cv.connectedComponentsWithStats(np.uint8(mask[:,:,z]), 4)[1:3]
#        print('++++++++++++++++++++++++')
#        print(labels)
#        print('------------------------')
#        print(stats)
#        print('++++++++++++++++++++++++')
        try:
            largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
            mask_part[labels == largest_label] = 1
        except:
            if verb:
                print('mask slice', z, 'is empty')
        temp_mask[:,:,z] = mask_part.astype(int)    
        
    new_mask = np.zeros_like(mask)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            for k in range(len(mask[i][j])):
                if mask[i][j][k] == 1:
                    new_mask[i-1][j-1][k] = 1
                    new_mask[i-1][j][k] = 1
                    new_mask[i-1][j+1][k] = 1
                    new_mask[i][j-1][k] = 1
                    new_mask[i][j][k] = 1
                    new_mask[i][j+1][k] = 1
                    new_mask[i+1][j-1][k] = 1
                    new_mask[i+1][j][k] = 1
                    new_mask[i+1][j+1][k] = 1
                    
    return new_mask

def display_brain_by_slice(brain, mask, patient, type):
    brain_max = brain.max()
    brain_min = brain.min()
    fig, axes = plt.subplots(5, 5, figsize=(20,20))
    fig.suptitle(patient + ' ' + type)
    for i in range(brain.shape[-1]):
        axes[i//5][i%5].imshow(brain[:,:,i].T, vmin=brain_min, vmax=brain_max, cmap='jet')
        axes[i//5][i%5].get_xaxis().set_visible(False)
        axes[i//5][i%5].get_yaxis().set_visible(False)
    plt.show()
    plt.close()
    