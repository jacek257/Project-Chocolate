'''
By: Jimi Cao, Austin Sophonsri

'''

import os
import subprocess
import argparse
import pandas as pd
import numpy as np
import json
from scipy import signal
import time
import glob
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import fnmatch

import stat_utils
import fft_analysis
import peak_analysis
import shifter
import parallel_processing

# instantiate the argument parser
parser = argparse.ArgumentParser()

# add positional argument
parser.add_argument("path", help="path of the folder that contains all the endtidal edit files")

# add optional arguments
parser.add_argument("-v", "--verbose", action='store_true', help="incrase output verbosity")
parser.add_argument("-p", "--plot", action='store_true', help='display the plots')
parser.add_argument("-f", "--fouier", action='store_true', help='switch analysis to fouier instead of default peak_find')
parser.add_argument("-b", "--block", action='store_true', help='switch analysis to block envelope')
parser.add_argument("-t", "--CO2_trough", action='store_true', help='switch CO2 peak finding to troughs')
parser.add_argument("-o", "--overwrite", action='store_true', help='overwrite existing processed gas data')
parser.add_argument("-m", "--manual", action='store_true', help='switch analysis to manual')
parser.add_argument("-s", "--sh", action='store_true', help='process SH cohort')
parser.add_argument("-w", "--wh", action='store_true', help='process WH cohort')
parser.add_argument("-i", "--invert", action='store_true', help='process failures by inverting O2 and CO2 detection and have')


#get the positional arguments
args = parser.parse_args()
path = args.path

verb = True if args.verbose else False
disp = True if args.plot else False
four = True if args.fouier else False
trough = True if args.CO2_trough else False
over = True if args.overwrite else False
block = True if args.block else False
man = True if args.manual else False
sh = True if args.sh else False
wh = True if args.wh else False
invert = True if args.invert else False

###########set directories (TODO, automate)
home_dir = '/media/ke/8tb_part2/FSL_work/'
if sh:
    nifti_dir = '/media/ke/8tb_part2/FSL_work/SH_info/'
    processed_dir = '/media/ke/8tb_part2/FSL_work/SH_info/BOLD_processed/'
    freesurfer_t1_dir = '/media/ke/8tb_part2/FSL_work/SH_FST1/'
elif wh:
    nifti_dir = '/media/ke/8tb_part2/FSL_work/WH_info/'
    processed_dir = '/media/ke/8tb_part2/FSL_work/WH_info/BOLD_processed/'
    freesurfer_t1_dir = '/media/ke/8tb_part2/FSL_work/WH_FST1/'
else:
    nifti_dir = '/media/ke/8tb_part2/FSL_work/all_info/'
    freesurfer_t1_dir = '/home/ke/Desktop/all_T1/'
    
feat_dir = '/media/ke/8tb_part2/FSL_work/feat/'
    

# set the limit for the number of processes (10 less that the total number of cores in the system) that can be run at once
cores = multiprocessing.cpu_count()
limit = cores - 5 if cores > 8 else 1
processes = [None] * limit

warnings = {'ID' : [],
            'warning' : []}

# make sure the path ends with '/'
if path[-1] != '/':
    path += '/'

# all grab all the .txt files in the endtidal folder
txt_files = [file for file in os.listdir(path) if file.upper().endswith('EDITS.TXT')]

if verb:
    print("Constructing dataframe for patients")
#separate patient ID and scan date and pack into dictionary
p_dic = {'ID' : [],
         'Date' : [],
         'EndTidal_Path' : [],
         'Notes' : []}
for f in txt_files:
    op = None
    ed = None
    notes = 'O2+CO2'
    
    for i in range(len(f)):
        if f[i] == '(':
            op = i
        if f[i] == ')':
            ed = i
            break
    
    if ed:
        f = f[:op] + f[ed+1:]
        notes = f[op:ed]
        
    file = f.split('_')
#    p_id = file[0].upper() + file[1]
    p_id = file[0].upper() if len(file[0]) > 4 else file[0].upper()+file[1]
    p_dic['ID'].append(p_id)
    p_dic['Date'].append(file[2])
    p_dic['EndTidal_Path'].append(path+f)
    p_dic['Notes'].append(notes)
    

p_df = pd.DataFrame(p_dic)

patient_BOLDS_header = [p_df.ID[i]+'_BOLD_'+p_df.Date[i] for i in range(len(p_df))]

if verb:
    print('Constructing dataframe that holds all relevant paths')
#get bold and FS paths
nii_paths = {'BOLD_path' : [], 'BOLD_corrected_path': [], 'T1_path' : [], 'meants_path': [], 'Processed_path': [], 'boldFS_exists': []}
for i in range(len(p_df)):
    if verb:
        print('\tGetting paths relavent to',p_df.ID[i] + '_' + p_df.Date[i])
    
    #if bold file doesnt exist then continue
    patient_dir = glob.glob(nifti_dir + p_df.ID[i] + '*' + p_df.Date[i])
    if len(patient_dir) == 0 or not os.path.exists(patient_dir[0] + '/BOLD/'):
        date = p_df.Date[i][-4:] + p_df.Date[i][:-4]
        patient_dir = glob.glob(nifti_dir + p_df.ID[i] + '*' + date)
        if len(patient_dir) == 0 or not os.path.exists(patient_dir[0] + '/BOLD/'):
            
            warnings['ID'].append(p_df.ID[i] + '_' + p_df.Date[i])
            warnings['warning'].append('No BOLD folder')
            
            nii_paths['BOLD_path'].append('')
            nii_paths['BOLD_corrected_path'].append('')
            nii_paths['T1_path'].append('')
            nii_paths['meants_path'].append('')
            nii_paths['Processed_path'].append('')
            nii_paths['boldFS_exists'].append(False)
            if verb:
                print('\t\tNo corresponding BOLD folder')
            continue
        else:
            p_df.Date[i] = date

    patient_dir = patient_dir[0] # patient dir is a list of len 1, need to actual string
    
    #get all matching files
    b_files = [file for file in os.listdir(patient_dir + '/BOLD/') if file.endswith('.nii')]
    fs_files = [file for file in os.listdir(freesurfer_t1_dir)  if fnmatch.fnmatch(file, p_df.ID[i]+'_'+p_df.Date[i]+'*_T1.nii*')]
    
    # check for the other naming convention
    if not fs_files:
        fs_files = [file for file in os.listdir(freesurfer_t1_dir) if fnmatch.fnmatch(file, p_df.ID[i]+'*_T1.nii*')]
    
    # use atlas from FSL
    if not fs_files:
        fs_files = ['/usr/local/fsl/data/standard/MNI152_T1_2mm_brain']
        warnings['ID'].append(p_df.ID[i] + '_' + p_df.Date[i])
        warnings['warning'].append('No FS_T1 file, using MNI152_T1_2mm_brain atlas')
        if verb:
            print('\t\tNo corresponding FS_T1 file, using MNI152_T1_2mm_brain atlas')
        

    #select and add file to appropriate list
    b_temp = patient_dir +'/BOLD/'+b_files[0] if len(b_files) > 0 else ''
    t_temp = freesurfer_t1_dir+fs_files[0] if len(fs_files) > 0 else ''

    nii_paths['BOLD_path'].append(b_temp)
    nii_paths['T1_path'].append(t_temp)
    nii_paths['boldFS_exists'].append(len(b_files) > 0 and len(fs_files)>0)

    if(len(b_files) > 0):
        #construct the processed nifti directory
        processed_dir = patient_dir +'/BOLD_processed/'
        
        if(not os.path.exists(processed_dir)):
            os.mkdir(processed_dir)

        #append processed directory to data frame
        nii_paths['Processed_path'].append(processed_dir)

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


        #add path of corrected nifti
        nii_paths['BOLD_corrected_path'].append(cor_temp)

        # extract the brain using BET
        brain_path = cor_temp[:-4]+'_brain.nii'
        if(not os.path.exists(brain_path+'.gz')):
            if verb:
                print('\t\tNo BET. Creating BET')
            subprocess.run(['bet', cor_temp, brain_path, '-F'])

        # get meants from BET
        meants_path = processed_dir+'/meants.txt'
        nii_paths['meants_path'].append(meants_path)
        if(not os.path.exists(meants_path)):
            if verb:
                print('\t\tNo meants. Creating meants')
            subprocess.run(['fslmeants', '-i', brain_path, '-o', meants_path])
    else:
        # warn people of the error
        warnings['ID'].append(p_df.ID[i] + '_' + p_df.Date[i])
        warnings['warning'].append('No BOLD file')
        
        nii_paths['Processed_path'].append('')
        nii_paths['BOLD_corrected_path'].append('')
        nii_paths['meants_path'].append('')
        if verb:
            print('\t\tNo corresponding BOLD file')
    if verb:
        print('\tAll relavent files grabbed')
        
if verb:
    print('Concatenating patient dataframe with path dataframe')
#append bold, FS paths, and conditional to p_df
p_df = pd.concat((p_df, pd.DataFrame(nii_paths)), axis=1)

if verb:
    print('Dropping patients that have missing BOLD')

#drop all false conditional rows and conditional column and reset indeces
p_df = p_df[p_df.boldFS_exists != False].drop('boldFS_exists', axis = 1)
p_df = p_df.reset_index(drop=True)


#get json files and grab TR value
if verb:
    print('Getting the TR for each patient')
tr_dict = {'ID' : [], 'Date': [], 'TR' : [], 'eff_TR' : [] }
for b_path in p_df.BOLD_path:
    p_id = b_path.split('/')[6].split('_')
    try:
        with open(b_path[:-4]+'.json', 'r') as j_file:
            data = json.load(j_file)
            tr_dict['ID'].append(p_id[0])
            tr_dict['Date'].append(p_id[2])
            tr = data['RepetitionTime']
            tr_dict['TR'].append(tr)
            eff_tr = tr * 2 if tr < 1.5 else tr
            tr_dict['eff_TR'].append(eff_tr)
    except:
        print('\t\t'+p_id[0]+'_'+p_id[2]+' has a bad json file. Cannot read')
        warnings['ID'].append(p_id[0]+'_'+p_id[2])
        warnings['warning'].append('Has a bad json file. Cannot read')
#exit()
#print(p_df)
#print(tr_dict)
p_df = p_df.merge(pd.DataFrame(tr_dict), on=['ID', 'Date'])

#get number of volumes
#run fslinfo and pipe -> python buffer -> replace \n with ' ' -> split into list -> choose correct index -> convert string to int
if verb:
    print('Getting the dimension of the BOLD for each patient with fslinfo')
p_df["Volumes"] = [int(subprocess.run(['fslinfo' ,b_path], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n',' ').split()[9]) for b_path in p_df.BOLD_path]

if verb:
    print('Checking total volume for each patient')
    
warning = p_df[p_df.TR != 1.5]
warning = warning.reset_index(drop=True)
for i in range(len(warning)):
#    print(i)
    warnings['ID'].append(warning.ID[i] + '_' + warning.Date[i])
    warnings['warning'].append('TR != 1.5')
    print('\t\t' + warning.iloc[i].ID + '_' + warning.iloc[i].Date + ' has a TR != 1.5')

####run EndTidal Cleaning and return paths

stats_df = pd.DataFrame()

# loop through different processing methods
#for typ in ['four', 'peak', 'trough', 'block']:
#for typ in ['four', 'peak', 'trough']:
#for typ in ['peak', 'trough']:
#
#for typ in ['four']:
for typ in ['block']:
    if typ == 'four':
        four = True
        trough = False
        block = False
    elif typ == 'trough':
        four = False
        trough = True
        block = False
    elif typ == 'block':
        four = False
        trough = False
        block = True
    else:
        four = False
        trough = False
        block = False

#if man:
    #generate cleaned data header
    if four:
        key = 'f_'
    elif trough:
        key = 't_'
    elif block:
        key = 'b_'
    elif man:
        key = 'm'
    else:
        key = 'p_'
    
    # create dictionary that will collect all relevant data
    ET_dict = {'ETO2' : [], 'ETCO2' : [], 'ET_exists' : [],
               'ID' : [], 'Date' : [], 'data_exists' : [],
               'O2_shift' : [], 'CO2_shift' : [], 'd_shift' : [], 'comb_shift' : [], 'O2_f_shift' : [],  'CO2_f_shift' : [], 
               'O2_r' : [], 'O2_p' : [], 'O2_f_r' : [], 'O2_f_p' : [],
               'CO2_r' : [], 'CO2_p' : [], 'CO2_f_r' : [], 'CO2_f_p' : [],
               'comb_r' : [], 'comb_p' : [], 'comb_f_r' : [], 'comb_f_p' : []
               }
    
    if verb:
        print('\n\nStart processing each patient')
    for f_path, vol, id, date, b_path, p_path, meants_path, tr in zip(p_df.EndTidal_Path, p_df.Volumes, p_df.ID, p_df.Date,
                                                                              p_df.BOLD_corrected_path, p_df.Processed_path, p_df.meants_path, p_df.eff_TR):
        if not os.path.exists(f_path[:-4]):
            os.mkdir(f_path[:-4])
    
        ##path made and stored in p_df.Processed_path
        ##pre-running slicetimer and mcflirt (motion correction) as we create bold paths
    
        #perfrom cleaning and load into p_df
        #load data into DataFrame
        endTidal = pd.read_csv(f_path, sep='\t|,', header=None, usecols=[0, 1, 2], index_col=False, engine='python')
        endTidal = endTidal.rename(columns={0 : 'Time',
                                            1 : 'O2',
                                            2 : 'CO2'})
        #drop rows with missing cols
        before = len(endTidal)
        endTidal = endTidal.dropna()
        after = len(endTidal)
        if before != after:
                warnings['ID'].append(id + '_' + date)
                warnings['warning'].append('dropped ' + str(before-after) + ' data points')
                print('\t\t' + id + '_' + date + ' has a dropped ' + str(before-after) + ' data points')
    
        #skip if DataFrame is empty
        if endTidal.empty:
            os.rmdir(f_path[:-4])
            ET_dict['ETO2'].append('')
            ET_dict['ETCO2'].append('')
            ET_dict['ET_exists'].append(False)
            ET_dict['data_exists'].append(False)
            print("\tpatient: ", id + '_' + date, "has empty end-tidal")
            continue
        
        # save dropped gas files if necessary
#        save_gas = '/home/ke/Desktop/all_dropped_gas/'
#        endTidal.to_csv(save_gas+id+'_'+date+f_path[-10:-4]+'.txt', index=False, header=False)
#        endTidal.to_excel(save_gas+id+'_'+date+f_path[-10:-4]+'.xlsx', index=False, header=False)
        
        scan_time = (vol-3) * tr
        
        try:
            meants = np.loadtxt(meants_path, delimiter='\n')[3:]
        except:
            print(id + '_' + date + ' cannot load meants')
        time_pts = np.arange(0, scan_time, tr)

        try:
            meants = signal.detrend(meants)
        except:
            print(id + '_' + date + ' cannot detrend meants')
            
        meants_df = pd.DataFrame({'Time' : time_pts,
                                  'Data' : meants})
        
        # save the bold df if necessary
        save_meants = '/home/ke/Desktop/all_meants/'
#        meants_df.to_csv(save_meants+id+'_'+date+f_path[-10:-4]+'.txt', index=False, header=False)
        meants_df.to_excel(save_meants+id+'_'+date+f_path[-10:-4]+'.xlsx', index=False, header=False)
#        continue
        
        CO2_only = False
        # Starting 2020, WH cohort was fed CO2 gas challenge only
        print(date)
        if (id[:2] == 'WH' and int(date) > 20200000) or id[:3] == 'CVR':
            CO2_only = True
    
        #generate cleaned data paths
        if not CO2_only:
            save_O2 = f_path[:-4]+'/'+key+'O2_contrast.txt'
        save_CO2 = f_path[:-4]+'/'+key+'CO2_contrast.txt'
        save = True
    
        #check if the save_files already exist
        if not CO2_only:
            if(os.path.exists(save_O2) and os.path.exists(save_CO2) and not over):
                save = False
                if(verb):
                    print('\tID: ', id + '_' + date," \tProcessed gas files already exist")
                ET_dict['ETO2'].append(save_O2)
                ET_dict['ETCO2'].append(save_CO2)
                ET_dict['ET_exists'].append(True)
                processed_O2 = np.loadtxt(save_O2)
                processed_CO2 = np.loadtxt(save_CO2)
        else:
            if(os.path.exists(save_CO2) and not over):
                save = False
                if(verb):
                    print('\tID: ', id + '_' + date," \tProcessed gas files already exist")
                ET_dict['ETO2'].append(None)
                ET_dict['ETCO2'].append(save_CO2)
                ET_dict['ET_exists'].append(True)
                processed_O2 = None
                processed_CO2 = np.loadtxt(save_CO2)
    
        # need to scale CO2 data is necessary
        if endTidal.CO2.max() < 1:
            endTidal.CO2 = endTidal.CO2 * 100
        
        
        if endTidal.Time.max() < 20:
            endTidal.Time = endTidal.Time * 60
        
        endTidal = endTidal[endTidal.Time < 600]
        if endTidal.Time.max() < 300:
            warnings['ID'].append(id + '_' + date)
            warnings['warning'].append('Not a full scan: ' + str(endTidal.Time.max()) + ' seconds. Cannot process')
            continue
        
        # begin storing relevant data that we want to keep after preprocessing
        ET_dict['ID'].append(id)
        ET_dict['Date'].append(date)
        ET_dict['data_exists'].append(True)
        
        i = 0
        diff = abs(endTidal.O2.iloc[i+1] - endTidal.O2.iloc[0])
        while diff < 1.2:
            i += 1
            diff = abs(endTidal.O2.iloc[i+1] - endTidal.O2.iloc[0])
        
        endTidal = endTidal[i:].reset_index(drop=True)
        
        i = len(endTidal)-1
        diff = abs(endTidal.O2.iloc[len(endTidal)-1] - endTidal.O2.iloc[i-1])
        while diff < 1.2:
            i -= 1
            diff = abs(endTidal.O2.iloc[len(endTidal)-1] - endTidal.O2.iloc[i-1])
        
        endTidal = endTidal[:i].reset_index(drop=True)

        # demean instead of detrend because detrend actually introduces a trend
        endTidal.CO2 = endTidal.CO2 - endTidal.CO2.mean()
        endTidal.O2 = endTidal.O2 - endTidal.O2.mean()
        
        if CO2_only:
            if verb:
                print('Starting CO2 analysis only for', id + '_' + date)
            pre_CO2 = peak_analysis.envelope(endTidal.Time, endTidal.CO2, tr, invert)
            pre_O2 = None
        elif four:
            if verb:
                print('Starting fourier for', id + '_' + date)
            # get fourier cleaned data
            # 1/60 because all gas challanges occur slower than at a rate of 1 per minute
            pre_O2 = fft_analysis.fourier_filter(endTidal.Time, endTidal.O2, 1/60, tr, time_pts, trim=True)
            pre_CO2 = fft_analysis.fourier_filter(endTidal.Time, endTidal.CO2, 1/60, tr, time_pts, trim=True)

        elif block:
            if verb:
                print('Starting block for', id + '_' + date)
            pre_O2 = peak_analysis.envelope(endTidal.Time, endTidal.O2.apply(lambda x:x*-1), tr, invert)
            pre_O2.Data *= -1
            pre_CO2 = peak_analysis.envelope(endTidal.Time, endTidal.CO2, tr, invert)

        elif trough:
            if verb:
                print('Starting troughs for', id + '_' + date)
            pre_CO2, pre_O2 = peak_analysis.peak_four(endTidal, verb, f_path, tr, time_pts, trough=True)
        
        elif man:
            if verb:
                print('Starting manual for ', id + '_' + date)
            pre_CO2, pre_O2 = peak_analysis.peak(endTidal, verb, f_path, time_pts)
        else:
            if verb:
                print('Starting peaks for', id + '_' + date)
            pre_CO2, pre_O2 = peak_analysis.peak_four(endTidal, verb, f_path, tr, time_pts, trough=False)

        
#        plt.plot(pre_O2.Data)
#        plt.show()
#        plt.plot(pre_CO2.Data)
#        plt.show()
#        exit()
        
        if pre_O2 is not None:
            if verb:
                print('Shifting O2')
            # get O2 shift   
            full_O2, processed_O2, O2_corr, O2_shift, O2_start = shifter.corr_align(meants, pre_O2.Time, pre_O2.Data, scan_time, time_pts, None, invert) # no ref_shift need for O2
    #        plt.plot(O2_corr)
    #        plt.show()
    #        plt.plot(processed_O2)
    #        plt.show()
            # get O2 r and p values
            O2_r, O2_p = st.pearsonr(processed_O2, meants)
        else:
            O2_start = None
        
        if verb:
            print('Shifting CO2')
        # get CO2 shift
        full_CO2, processed_CO2, CO2_corr, CO2_shift, CO2_start = shifter.corr_align(meants, pre_CO2.Time, pre_CO2.Data, scan_time, time_pts, O2_start, invert) # use O2 shift as ref for CO2 shift
#        plt.plot(CO2_corr)
#        plt.show()
#        plt.plot(processed_CO2)
#        plt.show()
        # get CO2 r and p values
        CO2_r, CO2_p = st.pearsonr(processed_CO2, meants)
        
        if pre_O2 is not None:
            if verb:
                print('Combining O2 and CO2')
            # combine the shfited O2 and CO2 and get the combined r and p values
            coeff, comb_r, comb_p = stat_utils.get_info([processed_O2, processed_CO2], meants)        
            combined = coeff[0] * processed_O2 + coeff[1] * processed_CO2 + coeff[2]
            if verb:
                print('Shifting Combination')
            # get the shift for the combination
            full_comb, processed_comb, comb_corr, comb_shift, comb_start = shifter.corr_align(meants, time_pts, combined, scan_time, time_pts, None, invert)
            # add the combined shift to O2 and CO2 shifts
            O2_f_shift = O2_shift + comb_shift
            CO2_f_shift = CO2_shift + comb_shift
    
            if verb:
                print('Interperolating original O2 and CO2 based on final shift')
            # interpolate the filtered O2 and CO2 data to fit the new time shifts      
            processed_O2_f = stat_utils.resamp(pre_O2.Time + O2_f_shift, time_pts, pre_O2.Data, O2_f_shift, O2_start+comb_start)
            processed_CO2_f = stat_utils.resamp(pre_CO2.Time + CO2_f_shift, time_pts, pre_CO2.Data, CO2_f_shift, CO2_start+comb_start)
            
            # get final O2 r and p values
            O2_f_r, O2_f_p = st.pearsonr(processed_O2_f, meants)
            # get final CO2 r and p values
            CO2_f_r, CO2_f_p = st.pearsonr(processed_CO2_f, meants)
        
            # get the combine the final shift signals and get their r and p values
            coeff_f, comb_f_r, comb_f_p = stat_utils.get_info([processed_O2_f, processed_CO2_f], meants)
        else:
            processed_CO2_f = processed_CO2
        
#        processed_O2_f, extend_time = analysis.stat_utils().resamp_f(pre_O2.Time + O2_f_shift, time_pts, pre_O2.Data, O2_f_shift, O2_start+comb_start, tr)
#        processed_CO2_f, extend_time = analysis.stat_utils().resamp_f(pre_CO2.Time + CO2_f_shift, time_pts, pre_CO2.Data, CO2_f_shift, CO2_start+comb_start, tr)

        # save the relevant information into the ET_dict
        ET_dict['O2_shift'].append(O2_shift if pre_O2 is not None else None)
        ET_dict['CO2_shift'].append(CO2_shift)
        ET_dict['d_shift'].append(CO2_shift-O2_shift if pre_O2 is not None else None)
        ET_dict['comb_shift'].append(comb_shift if pre_O2 is not None else None)
        ET_dict['O2_f_shift'].append(O2_f_shift if pre_O2 is not None else None)
        ET_dict['CO2_f_shift'].append(CO2_f_shift if pre_O2 is not None else None)
        
        ET_dict['O2_r'].append(O2_r if pre_O2 is not None else None)
        ET_dict['CO2_r'].append(CO2_r)
        ET_dict['comb_r'].append(comb_r if pre_O2 is not None else None)
        
        ET_dict['O2_p'].append(O2_p if pre_O2 is not None else None)
        ET_dict['CO2_p'].append(CO2_p)
        ET_dict['comb_p'].append(comb_p if pre_O2 is not None else None)
        
        ET_dict['O2_f_r'].append(O2_f_r if pre_O2 is not None else None)
        ET_dict['CO2_f_r'].append(CO2_f_r if pre_O2 is not None else None)
        ET_dict['comb_f_r'].append(comb_f_r if pre_O2 is not None else None)
        
        ET_dict['O2_f_p'].append(O2_f_p if pre_O2 is not None else None)
        ET_dict['CO2_f_p'].append(CO2_f_p if pre_O2 is not None else None)
        ET_dict['comb_f_p'].append(comb_f_p if pre_O2 is not None else None)
        
        
        if save:
            #storing cleaned data paths
            ET_dict['ETO2'].append(save_O2 if pre_O2 is not None else None)
            ET_dict['ETCO2'].append(save_CO2)
            ET_dict['ET_exists'].append(True)
            
            if verb:
                print('Saving')
            
            #save data
            if not CO2_only:
                np.savetxt(save_O2, processed_O2_f, delimiter='\t')
            np.savetxt(save_CO2, processed_CO2_f, delimiter='\t')
    
#            # save and create plots (shifts)
#            stat_utils.save_plots_comb_extend(df=endTidal, O2=pre_O2, O2_m=full_O2, O2_f=processed_O2_f, O2_corr=O2_corr,
#                                               CO2=pre_CO2, CO2_m=full_CO2, CO2_f=processed_CO2_f, CO2_corr=CO2_corr, meants=meants,
#                                               coeff=coeff, coeff_f=coeff_f, comb_corr=comb_corr, extend_time=extend_time,
#                                               f_path=f_path, key=key, verb=verb, time_points=time_pts, disp=disp)
            if CO2_only:
                stat_utils.save_plots_CO2(df=endTidal, CO2=pre_CO2, CO2_m=full_CO2, CO2_f=processed_CO2_f, CO2_corr=CO2_corr, meants=meants,
                                          f_path=f_path, key=key, verb=verb, time_points=time_pts, disp=disp)
            else:
                stat_utils.save_plots_comb(df=endTidal, O2=pre_O2, O2_m=full_O2, O2_f=processed_O2_f, O2_corr=O2_corr,
                                           CO2=pre_CO2, CO2_m=full_CO2, CO2_f=processed_CO2_f, CO2_corr=CO2_corr, meants=meants,
                                           coeff=coeff, coeff_f=coeff_f, comb_corr=comb_corr,
                                           f_path=f_path, key=key, verb=verb, time_points=time_pts, disp=disp)
    
    if verb:
        print()
            
    if verb:
        print('Finished processing each patient')
    
    if verb:
        print()
    
    #construct new DataFrame
    et_frame = pd.DataFrame(ET_dict)
    
    #merge and drop bad entries
    df = p_df.merge(et_frame, on=['ID', 'Date'], how='inner')
#    df = df[df.ET_exists != False].drop('ET_exists', axis = 1)
    
    #reset indeces
    df = df.reset_index(drop=True)
    
    #    print("df head----------------\n",p_df.head())
    
    if verb:
        print('\n\nStarting to run feat')
    #run Feat
    #check for (and make) feat directory
    if not os.path.exists(feat_dir):
        os.mkdir(feat_dir)
    
    #make design file directory
    if not os.path.exists(feat_dir+'design_files/'):
        os.mkdir(feat_dir+'design_files/')
    
    # load design template
    for i in range(len(df)):
        if df['O2_r'][i] is not None:
            with open(feat_dir+'design_files/template', 'r') as template:
                stringTemp = template.read()
                output_dir = feat_dir+key+df.ID[i]+'_'+df.Date[i]
    #            output_dir = '/media/ke/8tb_part2/FSL_work/feat/both_shift/'+key+df.ID[i]+'_'+df.Date[i]
                if os.path.exists(output_dir+'.feat'):
                    if verb:
                        print('FEAT already exists for', key+df.ID[i]+'_'+df.Date[i])
                    if over:
                        if verb:
                            print('Overwriting')
                        subprocess.run(['rm', '-rf', output_dir+'.feat'])
                    else:
                        continue
                to_write = stringTemp[:]
                # print(to_write)
                to_write = to_write.replace("%%OUTPUT_DIR%%",'"'+output_dir+'"')
                to_write = to_write.replace("%%VOLUMES%%",'"'+str(df.Volumes[i])+'"')
                to_write = to_write.replace("%%TR%%",'"'+str(df.eff_TR[i])+'"')
                to_write = to_write.replace("%%BOLD_FILE%%",'"'+df.BOLD_path[i]+'"')
                to_write = to_write.replace("%%FS_T1%%",'"'+df.T1_path[i]+'"')
                to_write = to_write.replace("%%O2_CONTRAST%%",'"'+df.ETO2[i]+'"')
                to_write = to_write.replace("%%CO2_CONTRAST%%",'"'+df.ETCO2[i]+'"')
        
                ds_path = feat_dir+'design_files/'+key+df.ID[i]+'_'+df.Date[i]+'.fsf'
                with open(ds_path, 'w+') as outFile:
                    outFile.write(to_write)
                            
                index = parallel_processing.get_next_avail(processes, verb, limit, key, 'FEAT')
                
                if verb:
                    print('Starting FEAT')
                processes[index] = subprocess.Popen(['feat', ds_path])
                time.sleep(0.5)
        else:
            with open(feat_dir+'design_files/template2', 'r') as template:
                stringTemp = template.read()
                output_dir = feat_dir+key+df.ID[i]+'_'+df.Date[i]
    #            output_dir = '/media/ke/8tb_part2/FSL_work/feat/both_shift/'+key+df.ID[i]+'_'+df.Date[i]
                if os.path.exists(output_dir+'.feat'):
                    if verb:
                        print('FEAT already exists for', key+df.ID[i]+'_'+df.Date[i])
                    if over:
                        if verb:
                            print('Overwriting')
                        subprocess.run(['rm', '-rf', output_dir+'.feat'])
                    else:
                        continue
                to_write = stringTemp[:]
                # print(to_write)
                to_write = to_write.replace("%%OUTPUT_DIR%%",'"'+output_dir+'"')
                to_write = to_write.replace("%%VOLUMES%%",'"'+str(df.Volumes[i])+'"')
                to_write = to_write.replace("%%TR%%",'"'+str(df.eff_TR[i])+'"')
                to_write = to_write.replace("%%BOLD_FILE%%",'"'+df.BOLD_path[i]+'"')
                to_write = to_write.replace("%%FS_T1%%",'"'+df.T1_path[i]+'"')
                to_write = to_write.replace("%%CO2_CONTRAST%%",'"'+df.ETCO2[i]+'"')
        
                ds_path = feat_dir+'design_files/'+key+df.ID[i]+'_'+df.Date[i]+'.fsf'
                with open(ds_path, 'w+') as outFile:
                    outFile.write(to_write)
                            
                index = parallel_processing.get_next_avail(processes, verb, limit, key, 'FEAT')
                
                if verb:
                    print('Starting FEAT')
                processes[index] = subprocess.Popen(['feat', ds_path])
                time.sleep(0.5)
            
    parallel_processing.wait_remaining(processes, verb, key, 'FEAT')
        
    # run featquery
    for i in range(len(df)):
        p_id = key+df.ID[i]+'_'+df.Date[i]
        feat_output_dir = feat_dir+p_id+'.feat/'
#        feat_output_dir = '/media/ke/8tb_part2/FSL_work/feat/both_shift/'+p_id+'.feat/'
        
        CO2_mask_dir_path = feat_output_dir+'cluster_mask_zstat1.nii.gz'
        O2_mask_dir_path = feat_output_dir+'cluster_mask_zstat2.nii.gz'
                    
        index = parallel_processing.get_next_avail(processes, verb, limit, key, 'featquery')
        
        if os.path.exists(feat_output_dir+'fq_CO2'):
            if verb:
                print('CO2 featquery already exists for', p_id)
            if over:
                if verb:
                    print('Overwriting')
                processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope1', 'fq_CO2', '-p', '-s', CO2_mask_dir_path])
        else:
            if verb:
                print('Starting CO2 featquery for', p_id)
            processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope1', 'fq_CO2', '-p', '-s', CO2_mask_dir_path])
        
        index = parallel_processing.get_next_avail(processes, verb, limit, key, 'featquery')
        
        if os.path.exists(feat_output_dir+'fq_O2'):
            if verb:
                print('O2 featquery already exists for', p_id)
            if over:
                if verb:
                    print('Overwriting')
                processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope2', 'fq_O2', '-p', '-s', O2_mask_dir_path])
        else:
            if verb:
                print('Starting featquery for O2')
            processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope2', 'fq_O2', '-p', '-s', O2_mask_dir_path])
    

    parallel_processing.wait_remaining(processes, verb, key, 'featquery')
        
    # get the stats
    for i in range(len(df)):        

        output_dir = feat_dir+key+df.ID[i]+'_'+df.Date[i]
#        output_dir = '/media/ke/8tb_part2/FSL_work/feat/both_shift/'+key+df.ID[i]+'_'+df.Date[i]
        feat_output_dir = output_dir+'.feat/'
        
        try:
            cz1 = pd.read_csv(feat_output_dir+'cluster_zstat1.txt', sep='\t', usecols=['Voxels', '-log10(P)', 'Z-MAX', 'COPE-MEAN'])
            t_vol = cz1.Voxels.sum()
            
            for j in range(len(cz1)):
                cz1.iloc[j] = cz1.iloc[j] * cz1.iloc[j].Voxels/t_vol
            

            z1 = { 'ID' : [df.ID[i]+'_'+df.Date[i]],
                   'type' : [key],
                   'Voxels': [t_vol],
                   '-log10(p)' : [cz1['-log10(P)'].sum()],
                   'COPE-MEAN' : [cz1['COPE-MEAN'].sum()]}
            cz1_final = pd.DataFrame(z1)
        
        except FileNotFoundError:
            warnings['ID'].append(df.ID[i] + '_' + df.Date[i])
            warnings['warning'].append('No cluster_zstat1.txt')
            if verb:
                print('No cluster_zstat1.txt')
                
            z1 = { 'ID' : [df.ID[i]+'_'+df.Date[i]],
                   'type' : [key],
                   'Voxels': [''],
                   '-log10(p)' : [''],
                   'COPE-MEAN' : ['']}
            cz1_final = pd.DataFrame(z1)
        
        try:
            cz2 = pd.read_csv(feat_output_dir+'cluster_zstat2.txt', sep='\t', usecols=['Voxels', '-log10(P)', 'Z-MAX', 'COPE-MEAN'])
            t_vol = cz2.Voxels.sum()
            
            for j in range(len(cz2)):
                cz2.iloc[j] = cz2.iloc[j] * cz2.iloc[j].Voxels/t_vol
            

            z2 = { 'ID' : [df.ID[i]+'_'+df.Date[i]],
                   'type' : [key],
                   'Voxels': [t_vol],
                   '-log10(p)' : [cz2['-log10(P)'].sum()],
                   'COPE-MEAN' : [cz2['COPE-MEAN'].sum()]}
            cz2_final = pd.DataFrame(z2)
        
        except FileNotFoundError:
            warnings['ID'].append(df.ID[i] + '_' + df.Date[i])
            warnings['warning'].append('No cluster_zstat2.txt')
            if verb:
                print('No cluster_zstat2.txt', df.ID[i], '_', df.Date[i])

            z2 = { 'ID' : [df.ID[i]+'_'+df.Date[i]],
                   'type' : [key],
                   'Voxels': [''],
                   '-log10(p)' : [''],
                   'COPE-MEAN' : ['']}
            cz2_final = pd.DataFrame(z2)
        
        build = cz1_final.merge(cz2_final, on=['ID', 'type'], suffixes=('_CO2', '_O2'))
        
        CO2_mask_dir_path = feat_output_dir+'cluster_mask_zstat1.nii.gz'
        O2_mask_dir_path = feat_output_dir+'cluster_mask_zstat2.nii.gz'
            
        O2 = feat_output_dir+'fq_O2/'
        try:
            fq1 = pd.read_csv(O2+'report.txt', sep='\t| ', header=None, usecols=[5], engine='python')
            fq1 = fq1.rename(columns={5 : 'fq_mean'})
            fq1['ID'] = df.ID[i]+'_'+df.Date[i]
            fq1['type'] = key
            fq1 = fq1[['ID', 'type', 'fq_mean']]
            build = build.merge(fq1, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
        except FileNotFoundError:
            warnings['ID'].append(df.ID[i] + '_' + df.Date[i])
            warnings['warning'].append('No O2 activation found')
            if verb:
                print('No O2 activation found for', df.ID[i] + '_' + df.Date[i])
            
            fq1 = pd.DataFrame({'ID' : [df.ID[i]+'_'+df.Date[i]],
                                'type' : [key],
                                'fq_mean' : ['']})
            build = build.merge(fq1, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
        
            
        CO2 = feat_output_dir+'fq_CO2/'
        try:
            fq2 = pd.read_csv(CO2+'report.txt', sep='\t| ', header=None, usecols=[5], engine='python')
            fq2 = fq2.rename(columns={5 : 'fq_mean'})
            fq2['ID'] = df.ID[i]+'_'+df.Date[i]
            fq2['type'] = key
            fq2 = fq2[['ID', 'type', 'fq_mean']]
            build = build.merge(fq2, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
        except FileNotFoundError:
            warnings['ID'].append(df.ID[i] + '_' + df.Date[i])
            warnings['warning'].append('No CO2 activation found')
            if verb:
                print('No CO2 activation found for', df.ID[i] + '_' + df.Date[i])
            
            fq2 = pd.DataFrame({'ID' : [df.ID[i]+'_'+df.Date[i]],
                                'type' : [key],
                                'fq_mean' : ['']})
            build = build.merge(fq2, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
        
        build['O2_shift'] = df.O2_shift[i]
        build['CO2_shift'] = df.CO2_shift[i]
        build['d_shift'] = df.d_shift[i]
        build['comb_shift'] = df.comb_shift[i]
        build['O2_f_shift'] = df.O2_f_shift[i]
        build['CO2_f_shift'] = df.CO2_f_shift[i]
        build['O2_r'] = df.O2_r[i]
        build['CO2_r'] = df.CO2_r[i]
        build['comb_r'] = df.comb_r[i]
        build['O2_p'] = df.O2_p[i]
        build['CO2_p'] = df.CO2_p[i]
        build['comb_p'] = df.comb_p[i]
        build['O2_f_r'] = df.O2_f_r[i]
        build['CO2_f_r'] = df.CO2_f_r[i]
        build['comb_f_r'] = df.comb_f_r[i]
        build['O2_f_p'] = df.O2_f_p[i]
        build['CO2_f_p'] = df.CO2_f_p[i]
        build['comb_f_p'] = df.comb_f_p[i]
#        print(build)

# Categories based on previous declarations
#ET_dict = {'ETO2' : [], 'ETCO2' : [], 'ET_exists' : [],
#           'ID' : [], 'Date' : [], 'data_exists' : [],
#           'O2_shift' : [], 'CO2_shift' : [], 'd_shift' : [], 'comb_shift' : [], 'O2_f_shift' : [],  'CO2_f_shift' : [], 
#           'O2_r' : [], 'O2_p' : [], 'O2_f_r' : [], 'O2_f_p' : [],
#           'CO2_r' : [], 'CO2_p' : [], 'CO2_f_r' : [], 'CO2_f_p' : [],
#           'comb_r' : [], 'comb_p' : [], 'comb_f_r' : [], 'comb_f_p' : []
#           }

#        build = pd.DataFrame({'ID' : [df.ID[i]+'_'+df.Date[i]],
#                              'O2_shift' : [df.O2_shift[i]],
#                              'CO2_shift' : [df.CO2_shift[i]],
#                              'd_shift' : [df.d_shift[i]],
#                              'O2_f_shift' : [df.O2_f_shift[i]],
#                              'CO2_f_shift' : [df.CO2_f_shift[i]],
#                              'O2_r' : [df.O2_r[i]],
#                              'CO2_r' : [df.CO2_r[i]],
#                              'comb_r' : [df.comb_r[i]],
#                              'O2_p' : [df.O2_p[i]],
#                              'CO2_p' : [df.CO2_p[i]],
#                              'comb_p' : [df.comb_p[i]],
#                              'O2_f_r' : [df.O2_f_r[i]],
#                              'CO2_f_r' : [df.CO2_f_r[i]],
#                              'comb_f_r' : [df.comb_f_r[i]],
#                              'O2_f_p' : [df.O2_f_p[i]],
#                              'CO2_f_p' : [df.CO2_f_p[i]],
#                              'comb_f_p' : [df.comb_f_p[i]],
#                              'data_exists' : [df.data_exists[i]]
#                              })
        
#        if add:
        stats_df = pd.concat([stats_df, build])
    
    if verb:
        print()
    
    stats_df.reset_index(drop=True)

warnings_df = pd.DataFrame(warnings).sort_values('ID')
#print(stats_df)
stats_df = stats_df.sort_values('ID')

f = plt.figure()
sns.set(rc={'figure.figsize':(30,20)})
plt.rc('legend', fontsize='medium')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='x-small')
plt.rc('axes', titlesize='medium')
        
with pd.ExcelWriter(path+'stats_data_comb.xlsx') as writer:  # doctest: +SKIP
    stats_df.to_excel(writer, sheet_name='Stats', index=False)
    warnings_df.to_excel(writer, sheet_name='Warnings', index=False)

sns.scatterplot(x='O2_f_shift', y='CO2_f_shift', data=stats_df)
sns.regplot(x='O2_f_shift', y='CO2_f_shift', data=stats_df)
slope, intercerpt, r_val, p_val, std_err = st.linregress(df.O2_f_shift, df.CO2_f_shift)
plt.text(-10, 50, 'R^2 = '+str(np.round(r_val**2, 4)))
plt.savefig(path+'shift_comp.png')
plt.show(block=False)
#plt.close()

f = plt.figure()
sns.distplot(stats_df.d_shift)
plt.savefig(path+'d_shift_dist.png')
plt.show(block=False)
#plt.close()
input('Press enter to end')

if verb:
    print('============== Script Finished ==============')


