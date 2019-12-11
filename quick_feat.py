#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:28:13 2019

@author: Jimi Cao
"""

import os
import subprocess
import pandas as pd
import time
import multiprocessing
import fnmatch
import analysis
import numpy as np
import json

# set limit for number of processes that can be run at once
cores = multiprocessing.cpu_count()
limit = cores - 5 if cores > 8 else 1
processes = [None] * limit
warnings = {'ID' : [],
            'warning' : []}

# grab the processed BOLD data, O2 contrast, and CO2 contrast
bold_dir = '/media/ke/8tb_part2/FSL_work/all_info/'
meants_dir = '/home/ke/Desktop/all_meants/'
shifted_dir = '/home/ke/Desktop/Matlab_output/'

# load design template
verb = True
key = ''
over = True

# grab all the contrast data
files = [file.split('_')[0]+'_'+file.split('_')[1] for file in os.listdir(shifted_dir) if file.upper().endswith('_O2.TXT')]
files.sort()
print(len(files))
O2_files = [file for file in os.listdir(shifted_dir) if file.upper().endswith('_O2.TXT')]
O2_files.sort()
print(len(O2_files))
CO2_files = [file for file in os.listdir(shifted_dir) if file.upper().endswith('_CO2.TXT')]
CO2_files.sort()
print(len(CO2_files))

# all grab all the .txt files in the bold folder
bold_files = []
for folder in os.listdir(bold_dir): 
    identify = folder.split('_')
    ID = identify[0] + '_' + identify[2]
    if ID not in files or not os.path.exists(bold_dir+folder+'/BOLD/'):
        identify = folder.split('_')
        warnings['ID'].append(ID)
        warnings['warning'].append('No BOLD folder')
        continue
    for file in os.listdir(bold_dir+folder+'/BOLD/'):
        if fnmatch.fnmatch(file, '*.nii'):
            bold_files.append(folder+'/BOLD/'+file)    
bold_files.sort()
print(len(bold_files))

meants_files = []
tr = []
for file in bold_files:
    #construct the processed nifti directory
    processed_dir = bold_dir+file.split('/')[0] +'/BOLD_processed/'
    
    if(not os.path.exists(processed_dir)):
        os.mkdir(processed_dir)
    
    b_temp = bold_dir + file
    try:
        with open(b_temp[:-4]+'.json', 'r') as j_file:
            data = json.load(j_file)
            rep_time = data['RepetitionTime']
            eff_tr = rep_time * 2 if rep_time < 1.5 else rep_time
            tr.append(eff_tr)
    except:
        print('\t\t'+file+' has a bad json file. Cannot read')
        
    #run slicetimer to correct for slice timing
    time_temp =  processed_dir+b_temp[b_temp.rfind('/')+1:-4]+'_timed.nii'
    if(not os.path.exists(time_temp+'.gz')):
        if verb:
            print('\t\tNo slice timing correction. Creating timing correction')
        subprocess.run(['slicetimer', '-i', b_temp, '-o', time_temp, '--odd'])
    #print('\n', cor_temp, '\n')

    # optional (motion correction)
    cor_temp = time_temp[:-4]+'_demotioned.nii'
    if(not os.path.exists(cor_temp+'.gz')):
        if verb:
            print('\t\tNo motion correction. Creating motion correction')
        subprocess.run(['mcflirt', '-in', time_temp, '-out', cor_temp])

    # extract the brain using BET
    brain_path = cor_temp[:-4]+'_brain.nii'
    if(not os.path.exists(brain_path+'.gz')):
        if verb:
            print('\t\tNo BET. Creating BET')
        subprocess.run(['bet', cor_temp, brain_path, '-F'])

    # get meants from BET
    meants_path = processed_dir+'meants.txt'
    if(not os.path.exists(meants_path)):
        if verb:
            print('\t\tNo meants. Creating meants')
        subprocess.run(['fslmeants', '-i', brain_path, '-o', meants_path])
    
#    print(meants_path)
    meants_files.append(meants_path)

df = pd.DataFrame({'meants' : meants_files,
                   'tr' : tr})
df = df.sort_values('meants')
print(len(df))
#input('Press enter to continue')

feat_dir = '/home/ke/Desktop/feat/'
T1_dir = '/home/ke/Desktop/all_T1/'

with open(feat_dir+'design_files/template', 'r') as template:
    stringTemp = template.read()
    for i in range(len(df)):
        meants_path = df['meants'][i]
        meants = np.loadtxt(meants_path, delimiter='\n')[3:]
        tr = df['tr'][1]
        identify = meants_path.split('/')[-3].split('_')
        sub_id = identify[0]
        date = identify[2]
        p_id = sub_id+'_'+date
        key = ''
        output_dir = feat_dir+p_id
        T1_file = [T1_dir+file for file in os.listdir(T1_dir) if fnmatch.fnmatch(file, sub_id+'_'+date+'*_T1.nii*')]
        
        if not T1_file:
            T1_file = [T1_dir+file for file in os.listdir(T1_dir) if fnmatch.fnmatch(file, sub_id+'_FS_T1.nii*')]
        
        if not T1_file:
            T1_file = ['/usr/local/fsl/data/standard/MNI152_T1_2mm_brain']
        
        T1_file = T1_file[0]
#         print(T1_file)
# #            output_dir = '/media/ke/8tb_part2/FSL_work/feat/both_shift/'+key+df.ID[i]+'_'+df.Date[i]
        if os.path.exists(output_dir+'.feat'):
            if verb:
                print('FEAT already exists for', p_id)
            if over:
                if verb:
                    print('Overwriting')
                subprocess.run(['rm', '-rf', output_dir+'.feat'])
            else:
                continue
        to_write = stringTemp[:]
       # print(to_write)
        to_write = to_write.replace("%%OUTPUT_DIR%%",'"'+output_dir+'"')
        to_write = to_write.replace("%%VOLUMES%%",'"'+str(len(meants)+3)+'"')
        to_write = to_write.replace("%%TR%%",'"'+str(tr)+'"')
        to_write = to_write.replace("%%BOLD_FILE%%",'"'+bold_dir+bold_files[i]+'"')
        to_write = to_write.replace("%%FS_T1%%",'"'+T1_file+'"')
        to_write = to_write.replace("%%O2_CONTRAST%%",'"'+shifted_dir+O2_files[i]+'"')
        to_write = to_write.replace("%%CO2_CONTRAST%%",'"'+shifted_dir+CO2_files[i]+'"')

        ds_path = feat_dir+'design_files/'+p_id+'.fsf'
        
        with open(ds_path, 'w+') as outFile:
            outFile.write(to_write)

        index = analysis.parallel_processing().get_next_avail(processes, verb, limit, key, 'FEAT')

        if verb:
            print('Starting FEAT')
        processes[index] = subprocess.Popen(['feat', ds_path])
        time.sleep(0.5)

    analysis.parallel_processing().wait_remaining(processes, verb, key, 'FEAT')

# run featquery
for i in range(len(bold_files)):
    identify = bold_files[i].split('_')
#    print(identify)
    sub_id = identify[0]
    date = identify[2].split('/')[0]
    p_id = sub_id+'_'+date
#    print(p_id)
    key = ''
    feat_output_dir = feat_dir+p_id+'.feat/'
#        feat_output_dir = '/media/ke/8tb_part2/FSL_work/feat/both_shift/'+p_id+'.feat/'
#    print(feat_output_dir)
#    exit()

    O2_mask_dir_path = feat_output_dir+'cluster_mask_zstat1.nii.gz'
    CO2_mask_dir_path = feat_output_dir+'cluster_mask_zstat2.nii.gz'

    index = analysis.parallel_processing().get_next_avail(processes, verb, limit, key, 'featquery')

    if os.path.exists(feat_output_dir+'fq_O2'):
        if verb:
            print('O2 featquery already exists for', p_id)
        if over:
            if verb:
                print('Overwriting')
            os.system('rm -rf '+feat_output_dir+'fq_O2')
            processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope1', 'fq_O2', '-p', '-s', O2_mask_dir_path])
    else:
        if verb:
            print('Starting O2 featquery for', p_id)
        processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope1', 'fq_O2', '-p', '-s', O2_mask_dir_path])

    index = analysis.parallel_processing().get_next_avail(processes, verb, limit, key, 'featquery')

    if os.path.exists(feat_output_dir+'fq_CO2'):
        if verb:
            print('CO2 featquery already exists for', p_id)
        if over:
            if verb:
                print('Overwriting')
            os.system('rm -rf '+feat_output_dir+'fq_CO2')
            processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope2', 'fq_CO2', '-p', '-s', CO2_mask_dir_path])
    else:
        if verb:
            print('Starting featquery for CO2')
        processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope2', 'fq_CO2', '-p', '-s', CO2_mask_dir_path])


analysis.parallel_processing().wait_remaining(processes, verb, key, 'featquery')

stats_df = pd.DataFrame()
# get the stats
for i in range(len(bold_files)):
    identify = bold_files[i].split('_')
    sub_id = identify[0]
    date = identify[2].split('/')[0]
    p_id = sub_id+'_'+date
    key = ''
    feat_output_dir = feat_dir+p_id+'.feat/'
#        output_dir = '/media/ke/8tb_part2/FSL_work/feat/both_shift/'+key+df.ID[i]+'_'+df.Date[i]
#     feat_output_dir = output_dir+'.feat/'

    try:
        cz1 = pd.read_csv(feat_output_dir+'cluster_zstat1.txt', sep='\t', usecols=['Voxels', '-log10(P)', 'Z-MAX', 'COPE-MEAN'])
        t_vol = cz1.Voxels.sum()

        for j in range(len(cz1)):
            cz1.iloc[j] = cz1.iloc[j] * cz1.iloc[j].Voxels/t_vol


        z1 = { 'ID' : [p_id],
               'Voxels': [t_vol],
               '-log10(p)' : [np.round(cz1['-log10(P)'].sum(), 2)],
               'COPE-MEAN' : [np.round(cz1['COPE-MEAN'].sum(), 2)]}
        cz1_final = pd.DataFrame(z1)

    except FileNotFoundError:
        warnings['ID'].append(p_id)
        warnings['warning'].append('No cluster_zstat1.txt')
        if verb:
            print('No cluster_zstat1.txt', p_id)

        z1 = { 'ID' : [p_id],
               'Voxels': [''],
               '-log10(p)' : [''],
               'COPE-MEAN' : ['']}
        cz1_final = pd.DataFrame(z1)

    try:
        cz2 = pd.read_csv(feat_output_dir+'cluster_zstat2.txt', sep='\t', usecols=['Voxels', '-log10(P)', 'Z-MAX', 'COPE-MEAN'])
        t_vol = cz2.Voxels.sum()

        for j in range(len(cz2)):
            cz2.iloc[j] = cz2.iloc[j] * cz2.iloc[j].Voxels/t_vol


        z2 = { 'ID' : [p_id],
               'Voxels': [t_vol],
               '-log10(p)' : [np.round(cz2['-log10(P)'].sum(), 2)],
               'COPE-MEAN' : [np.round(cz2['COPE-MEAN'].sum(), 2)]}
        cz2_final = pd.DataFrame(z2)

    except FileNotFoundError:
        warnings['ID'].append(p_id)
        warnings['warning'].append('No cluster_zstat2.txt')
        if verb:
            print('No cluster_zstat2.txt', p_id)

        z2 = { 'ID' : [p_id],
               'Voxels': [''],
               '-log10(p)' : [''],
               'COPE-MEAN' : ['']}
        cz2_final = pd.DataFrame(z2)

    build = cz1_final.merge(cz2_final, on=['ID'], suffixes=('_O2', '_CO2'))

    O2_mask_dir_path = feat_output_dir+'cluster_mask_zstat1.nii.gz'
    CO2_mask_dir_path = feat_output_dir+'cluster_mask_zstat2.nii.gz'

    O2 = feat_output_dir+'fq_O2/'
    try:
        fq1 = pd.read_csv(O2+'report.txt', sep='\t| ', header=None, usecols=[5], engine='python')
        fq1 = fq1.rename(columns={5 : 'fq_mean'})
        fq1['ID'] = p_id
        fq1 = fq1[['ID', 'fq_mean']]
        build = build.merge(fq1, on=['ID'], suffixes=('_O2', '_CO2'))
    except FileNotFoundError:
        warnings['ID'].append(p_id)
        warnings['warning'].append('No O2 activation found')
        if verb:
            print('No O2 activation found for', p_id, 'O2')
            
        fq1 = pd.DataFrame({'ID' : [p_id],
                            'fq_mean' : ['']})
        build = build.merge(fq1, on=['ID', 'type'], suffixes=('_O2', '_CO2'))


    CO2 = feat_output_dir+'fq_CO2/'
    try:
        fq2 = pd.read_csv(CO2+'report.txt', sep='\t| ', header=None, usecols=[5], engine='python')
        fq2 = fq2.rename(columns={5 : 'fq_mean'})
        fq2['ID'] = p_id
        fq2 = fq2[['ID', 'fq_mean']]
        build = build.merge(fq2, on=['ID'], suffixes=('_O2', '_CO2'))
    except FileNotFoundError:
        warnings['ID'].append(p_id)
        warnings['warning'].append('No CO2 activation found')
        if verb:
            print('No CO2 activation found for', p_id, 'CO2')
            
        fq2 = pd.DataFrame({'ID' : [p_id],
                            'fq_mean' : ['']})
        build = build.merge(fq2, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
    
    stats_df = pd.concat([stats_df, build])

#     build['O2_shift'] = df.O2_shift[i]
#     build['CO2_shift'] = df.CO2_shift[i]
#     build['O2_coeff'] = df.coeffs[i][0]
#     build['CO2_coeff'] = df.coeffs[i][1]
#     build['r'] = df.r[i]
#     build['p_value'] = df.p_value[i]
            
warnings_df = pd.DataFrame(warnings).sort_values('ID')
with pd.ExcelWriter(shifted_dir+'stats_data.xlsx') as writer:  # doctest: +SKIP
    stats_df.to_excel(writer, sheet_name='Stats', index=False)
    warnings_df.to_excel(writer, sheet_name='Warnings', index=False)