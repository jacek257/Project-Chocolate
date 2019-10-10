'''
By: Jimi Cao, Austin Sophonsri

'''

import os
import subprocess
import argparse
import pandas as pd
import numpy as np
import analysis
import json
from scipy import signal
import time
import glob
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import fnmatch

# instantiate the argument parser
parser = argparse.ArgumentParser()

# add positional argument
parser.add_argument("path", help="path of the folder that contains all the endtidal edit files")

# add optional arguments
parser.add_argument("-v", "--verbose", action='store_true', help="incrase output verbosity")
parser.add_argument("-f", "--fouier", action='store_true', help='switch analysis to fouier instead of default peak_find')
parser.add_argument("-b", "--block", action='store_true', help='switch analysis to block envelope')
parser.add_argument("-t", "--CO2_trough", action='store_true', help='switch CO2 peak finding to troughs')
parser.add_argument("-o", "--overwrite", action='store_true', help='overwrite existing processed gas data')
parser.add_argument("-m", "--manual", action='store_true', help='switch analysis to manual')
parser.add_argument("-w", "--wh", action='store_true', help='process WH cohort')


#get the positional arguments
args = parser.parse_args()
path = args.path

verb = True if args.verbose else False
four = True if args.fouier else False
trough = True if args.CO2_trough else False
over = True if args.overwrite else False
block = True if args.block else False
man = True if args.manual else False
wh = True if args.wh else False

###########set directories (TODO, automate)
home_dir = '/media/ke/8tb_part2/FSL_work/'
if not wh:
    nifti_dir = '/media/ke/8tb_part2/FSL_work/SH_info/'
    processed_dir = '/media/ke/8tb_part2/FSL_work/SH_info/BOLD_processed'
    freesurfer_t1_dir = '/media/ke/8tb_part2/FSL_work/SH_FST1/'
else:
    nifti_dir = '/media/ke/8tb_part2/FSL_work/WH_info/'
    processed_dir = '/media/ke/8tb_part2/FSL_work/WH_info/BOLD_processed'
    freesurfer_t1_dir = '/media/ke/8tb_part2/FSL_work/WH_FST1/'
    
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
p_dic = {'Cohort' : [],
         'ID' : [],
         'Date' : [],
         'EndTidal_Path' : []}
for f in txt_files:
    file = f.split('_')
    p_dic['Cohort'].append(file[0].upper())
    p_dic['ID'].append(file[1])
    p_dic['Date'].append(file[2])
    p_dic['EndTidal_Path'].append(path+f)

p_df = pd.DataFrame(p_dic)
#p_df = pd.DataFrame({
#                'Cohort':[f[0:2] for f in txt_files],
#                'ID':[f[3:6] for f in txt_files],
#                'Month':[f[11:13] for f in txt_files],
#                'Day':[f[13:15] for f in txt_files],
#                'Year':[f[7:11] for f in txt_files],
#                'EndTidal_Path': [path+f for f in txt_files]
#             })


#print(p_df.head())
#create patient bold scan listdir (is a list not DataFrame)
#patient_BOLDS_header = [p_df.Cohort[i]+p_df.ID[i]+'_BOLD_'+p_df.Year[i]+p_df.Month[i]+p_df.Day[i]
#                    for i in range(len(p_df))]
patient_BOLDS_header = [p_df.Cohort[i]+p_df.ID[i]+'_BOLD_'+p_df.Date[i]
                    for i in range(len(p_df))]

if verb:
    print('Constructing dataframe that holds all relevant paths')
#get bold and FS paths
nii_paths = {'BOLD_path' : [], 'BOLD_corrected_path': [], 'T1_path' : [], 'meants_path': [], 'Processed_path': [], 'boldFS_exists': []}
for i in range(len(p_df)):
    if verb:
        print('\tGetting paths relavent to',p_df.Cohort[i] + p_df.ID[i] + '_' + p_df.Date[i])
    
    #if bold file doesnt exist then continue
    patient_dir = glob.glob(nifti_dir + p_df.Cohort[i] + p_df.ID[i] + '*' + p_df.Date[i])
    if len(patient_dir) == 0 or not os.path.exists(patient_dir[0] + '/BOLD/'):
        date = p_df.Date[i][-4:] + p_df.Date[i][:-4]
        patient_dir = glob.glob(nifti_dir + p_df.Cohort[i] + p_df.ID[i] + '*' + date)
        if len(patient_dir) == 0 or not os.path.exists(patient_dir[0] + '/BOLD/'):
            
            warnings['ID'].append(p_df.Cohort[i] + p_df.ID[i] + '_' + p_df.Date[i])
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
    
#    fs_files = [file for file in os.listdir(freesurfer_t1_dir) if file == p_df.Cohort[i]+p_df.ID[i]+'_FS_T1.nii.gz']
#    print(p_df.Cohort[i]+'*'+p_df.ID[i]+'_'+p_df.Date[i]+'_FS_TI.nii*')
    fs_files = [file for file in os.listdir(freesurfer_t1_dir)  if fnmatch.fnmatch(file, p_df.Cohort[i]+'*'+p_df.ID[i]+'_'+p_df.Date[i]+'_FS_T1.nii*')]
#    for file in os.listdir(freesurfer_t1_dir):
#        print(file)
#    exit()
    if not fs_files:
        fs_files = [file for file in os.listdir(freesurfer_t1_dir) if fnmatch.fnmatch(file, p_df.Cohort[i]+p_df.ID[i]+'_FS_T1.nii*')]

    #select and add file to appropriate list
    b_temp = patient_dir +'/BOLD/'+b_files[0] if len(b_files) > 0 else ''
    t_temp = freesurfer_t1_dir+fs_files[0] if len(fs_files) > 0 else ''

    nii_paths['BOLD_path'].append(b_temp)
    nii_paths['T1_path'].append(t_temp)
    nii_paths['boldFS_exists'].append(len(b_files) > 0 and len(fs_files)>0)
    
#    print(fs_files)
    
    if len(fs_files) == 0:
        warnings['ID'].append(p_df.Cohort[i] + p_df.ID[i] + '_' + p_df.Date[i])
        warnings['warning'].append('No FS_T1 file')
        if verb:
            print('\t\tNo corresponding FS_T1 file')

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
        warnings['ID'].append(p_df.Cohort[i] + p_df.ID[i] + '_' + p_df.Date[i])
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


if verb:
    print('Getting the TR for each patient')
#get json files and read CSV
tr_dict = {'TR' : [], 'eff_TR' : [] }

for b_path in p_df.BOLD_path:
#    print(b_path[:-4]+'.json')
    with open(b_path[:-4]+'.json', 'r') as j_file:
        data = json.load(j_file)
        #print(data['RepetitionTime'])
        tr = data['RepetitionTime']
        tr_dict['TR'].append(tr)
        eff_tr = tr * 2 if tr < 1.5 else 1.5
        tr_dict['eff_TR'].append(eff_tr)

p_df = pd.concat((p_df, pd.DataFrame(tr_dict)), axis=1)
#print('\n',p_df)

#get number of volumes
#run fslinfo and pipe -> python buffer -> replace \n with ' ' -> split into list -> choose correct index -> convert string to int
if verb:
    print('Getting the dimension of the BOLD for each patient with fslinfo')
p_df["Volumes"] = [int(subprocess.run(['fslinfo' ,b_path], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n',' ').split()[9]) for b_path in p_df.BOLD_path]

if verb:
    print('Checking total volume for each patient')
    
warning = p_df[p_df.TR != 1.5]
warning = warning.reset_index(drop=True)
#print(warning)
for i in range(len(warning)):
#    print(i)
    warnings['ID'].append(warning.Cohort[i] + warning.ID[i] + '_' + warning.Date[i])
    warnings['warning'].append('TR != 1.5')
    print('\t\t' + warning.iloc[i].Cohort + warning.iloc[i].ID + '_' + warning.iloc[i].Date + ' has a TR != 1.5')
    
p_df = p_df[p_df.TR == 1.5]

#choose non-trivial bold series
#warning_2 = p_df[p_df.Volumes < 150]
#for i in range(len(warning_2)):
#    print('\t\t' + warning_2[i].Cohort + warning_2[i].ID + '_' + warning_2[i].Date + 'has a total volume < 150')
#print('\np_df w/ dim > 150\n',p_df.head())

####run EndTidal Cleaning and return paths

stats_df = pd.DataFrame()

#for smoothing in ['none', 'pre', 'post']:
#for smoothing in ['no', 'pre']:
#    if smoothing == 'pre':
#        pre = True
#        post = False
#        sm = 'pre_'
#    else:
#        pre = False
#        post = False
#        sm = "no_"
    # typ = 'block'
    # if typ:
#for typ in ['four', 'peak', 'trough', 'block']:
#for typ in ['four', 'peak', 'trough']:
#for typ in ['peak', 'trough']:
for typ in ['four']:
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
    
    ET_dict = {'ETO2' : [], 'ETCO2' : [], 'ET_exists' : [], 'Cohort' : [], 'ID' : [], 'Date' : [],
               'O2_shift' : [], 'CO2_shift' : [], 'coeffs' : [], 'r' : [], 'p_value' : []}
    
    to_drop = []
    
    if verb:
        print('\n\nStart processing each patient')
    for f_path, vol, cohort, id, date, b_path, p_path, meants_path, tr in zip(p_df.EndTidal_Path, p_df.Volumes, p_df.Cohort, p_df.ID, p_df.Date,
                                                                              p_df.BOLD_corrected_path, p_df.Processed_path, p_df.meants_path, p_df.eff_TR):
        if not os.path.exists(f_path[:-4]):
            os.mkdir(f_path[:-4])
    
        ##path made and stored in p_df.Processed_path
        ##pre-running slicetimer and mcflirt (motion correction) as we create bold paths
    
        #perfrom cleaning and load into p_df
        #load data into DataFrame
        ET_dict['Cohort'].append(cohort)
        ET_dict['ID'].append(id)
        ET_dict['Date'].append(date)
        endTidal = pd.read_csv(f_path, sep='\t|,', header=None, usecols=[0, 1, 2], index_col=False, engine='python')
        endTidal = endTidal.rename(columns={0 : 'Time',
                                            1 : 'O2',
                                            2 : 'CO2'})
        #drop rows with missing cols
        endTidal = endTidal.dropna()
        
    
        #skip if DataFrame is empty
        if endTidal.empty:
            os.rmdir(f_path[:-4])
            ET_dict['ETO2'].append('')
            ET_dict['ETCO2'].append('')
            ET_dict['ET_exists'].append(False)
            print("\tpatient: ", cohort + id + '_' + date, "has empty end-tidal")
            continue
        
        scan_time = (vol-3) * tr
    
        meants = np.loadtxt(meants_path, delimiter='\n')[3:]
        time_pts = np.arange(0, scan_time, tr)
    
        #generate cleaned data paths
        save_O2 = f_path[:-4]+'/'+key+'O2_contrast.txt'
        save_CO2 = f_path[:-4]+'/'+key+'CO2_contrast.txt'
        save = True
    
        #check if the save_files already exist
        if(os.path.exists(save_O2) and os.path.exists(save_CO2) and not over):
            save = False
            if(verb):
                print('\tID: ', cohort + id + '_' + date," \tProcessed gas files already exist")
            ET_dict['ETO2'].append(save_O2)
            ET_dict['ETCO2'].append(save_CO2)
            ET_dict['ET_exists'].append(True)
            processed_O2 = np.loadtxt(save_O2)
            processed_CO2 = np.loadtxt(save_CO2)
    
        # need to scale CO2 data is necessary
        if endTidal.CO2.max() < 1:
            endTidal.CO2 = endTidal.CO2 * 100
        
#        plt.plot(endTidal.CO2)
#        plt.show()
        
        if endTidal.Time.max() < 20:
            endTidal.Time = endTidal.Time * 60
        
        i = 1
        diff = abs(endTidal.O2.iloc[i+1] - endTidal.O2.iloc[0])
        while diff < 2:
            i += 1
            diff = abs(endTidal.O2.iloc[i+1] - endTidal.O2.iloc[0])
        
        endTidal = endTidal[i:].reset_index(drop=True)
        
        i = len(endTidal)-2
        diff = abs(endTidal.O2.iloc[len(endTidal)-1] - endTidal.O2.iloc[i])
        while diff < 2:
            i -= 1
            diff = abs(endTidal.O2.iloc[len(endTidal)-1] - endTidal.O2.iloc[i])
        
        endTidal = endTidal[:i].reset_index(drop=True)
        
        meants = signal.savgol_filter(meants, 5, 3)
        
        endTidal.CO2 = signal.savgol_filter(endTidal.CO2, 35, 3)
        endTidal.CO2 = signal.detrend(endTidal.CO2)
        endTidal.O2 = signal.detrend(endTidal.O2)
        
#        plt.plot(endTidal.CO2)
#        plt.show()
#        exit()
#            print(endTidal)

        if four:
            if verb:
                print('Starting fourier for', cohort + id + '_' + date)
            #get fourier cleaned data
            pre_O2 = analysis.fft_analysis().fourier_filter(endTidal.Time, endTidal.O2, 3/60, 25/60, tr, time_pts)
            pre_O2.Data = signal.savgol_filter(pre_O2.Data, 5, 3)
            pre_CO2 = analysis.fft_analysis().fourier_filter(endTidal.Time, endTidal.CO2, 3/60, 25/60, tr, time_pts)
            pre_CO2.Data = signal.savgol_filter(pre_CO2.Data, 5, 3)
        elif block:
            if verb:
                print('Starting block for', cohort + id + '_' + date)
            pre_O2 = analysis.peak_analysis().block_signal(endTidal.Time, endTidal.O2.apply(lambda x:x*-1), time_pts)*-1
            pre_CO2 = analysis.peak_analysis().block_signal(endTidal.Time, endTidal.CO2, time_pts)

        elif trough:
            if verb:
                print('Starting troughs for', cohort + id + '_' + date)
            pre_CO2, pre_O2 = analysis.peak_analysis().peak_four(endTidal, verb, f_path, tr, time_pts, trough=True)
        
        elif man:
            if verb:
                print('Starting manual for ', cohort + id + '_' + date)
            pre_CO2, pre_O2 = analysis.peak_analysis().peak(endTidal, verb, f_path, time_pts)
        else:
            if verb:
                print('Starting peaks for', cohort + id + '_' + date)
            pre_CO2, pre_O2 = analysis.peak_analysis().peak_four(endTidal, verb, f_path, tr, time_pts, trough=False)

        # get shifted O2 and CO2
#        all_CO2.Data -= all_CO2.Data.mean()
#        all_CO2.Data /- all_CO2.Data.std()
#        
#        all_CO2.meants -= all_CO2.meants.mean()
#        all_CO2.meants /- all_CO2.meants.std()
#        
#        
#        r = st.pearsonr(all_CO2.meants , all_CO2.Data)
#        print(r)
#        corr = signal.correlate(all_CO2.meants, all_CO2.Data)
#        plt.plot(corr/corr.std())
#        plt.show()
#
#        sns.lineplot(x='Time', y='Data', data=all_CO2)
#        sns.lineplot(x='Time', y='meants', data=all_CO2)
#        plt.show()
#        exit()
        
        full_O2, processed_O2, O2_corr, O2_shift, O2_start = analysis.shifter().corr_align(meants, pre_O2.Time, pre_O2.Data, scan_time, time_pts)
#        O2_shift = comb_shift
#        ET_dict['O2_shift'].append(O2_shift)
        full_CO2, processed_CO2, CO2_corr, CO2_shift, CO2_start = analysis.shifter().corr_align(meants, pre_CO2.Time, pre_CO2.Data, scan_time, time_pts)
#        CO2_shift = comb_shift
#        ET_dict['CO2_shift'].append(CO2_shift)
        
#        processed_comb = processed_O2 * processed_CO2
        
#        full_O2.to_excel(id+'_'+key+'all_O2.xlsx', index=False)
#        full_CO2.to_excel(id+'_'+key+'all_CO2.xlsx', index=False)
        coeff, r, p_value = analysis.stat_utils().get_info([processed_O2, processed_CO2], meants)
        
        combined = coeff[0] * processed_O2 + coeff[1] * processed_CO2 + coeff[2]
        full_comb, processed_comb, comb_corr, comb_shift, comb_start = analysis.shifter().corr_align(meants, time_pts, combined, scan_time, time_pts)
        O2_shift += comb_shift
        CO2_shift += comb_shift
        
        processed_O2_f = analysis.stat_utils().resamp(pre_O2.Time + O2_shift, time_pts, pre_O2.Data, O2_shift, O2_start+comb_start)
        processed_CO2_f = analysis.stat_utils().resamp(pre_CO2.Time + CO2_shift, time_pts, pre_CO2.Data, CO2_shift, CO2_start+comb_start)
        
        coeff_f, r, p_value = analysis.stat_utils().get_info([processed_O2_f, processed_CO2_f], meants)
        
#        trim_O2 = pre_O2.Data[:len(meants)]
#        tirm_CO2 = pre_CO2.Data[:len(meants)]
#        coeff, r, p_value = analysis.stat_utils().get_info([trim_O2, tirm_CO2], meants)
#        
#        comb_data = coeff[0] * pre_O2.Data + coeff[1] * pre_CO2.Data + coeff[2]
#        comb_time = pre_O2.Time
#        full_comb, processed_comb, comb_corr, comb_shift, comb_idx = analysis.shifter().corr_align(meants, comb_time, comb_data, scan_time, time_pts)
#        O2_shift = comb_shift
#        O2_corr = comb_corr
#        CO2_shift = comb_shift
#        CO2_corr = comb_corr
#        
#        processed_O2 = analysis.stat_utils().resamp(pre_O2.Time + O2_shift, time_pts, pre_O2.Data, O2_shift, comb_idx)
#        processed_CO2 = analysis.stat_utils().resamp(pre_CO2.Time + CO2_shift, time_pts, pre_CO2.Data, CO2_shift, comb_idx)        

        ET_dict['O2_shift'].append(O2_shift)
        ET_dict['CO2_shift'].append(CO2_shift)
        
#        r, p_value = st.pearsonr(coeff[0] * processed_O2 + coeff[1] * processed_CO2 + coeff[2], meants)
        
#        ET_dict['coeffs'].append(coeff)
        ET_dict['coeffs'].append(coeff_f)
        ET_dict['r'].append(r)
        ET_dict['p_value'].append(p_value)
        
        
        if save:
            #storing cleaned data paths
            ET_dict['ETO2'].append(save_O2)
            ET_dict['ETCO2'].append(save_CO2)
            ET_dict['ET_exists'].append(True)
    
#            processed_O2 = signal.savgol_filter(processed_O2, 11, 3)
#            processed_CO2 = signal.savgol_filter(processed_CO2, 11, 3)
            
            #save data
#            np.savetxt(save_O2, processed_O2, delimiter='\t')
#            np.savetxt(save_CO2, processed_CO2, delimiter='\t')
            np.savetxt(save_O2, processed_O2_f, delimiter='\t')
            np.savetxt(save_CO2, processed_CO2_f, delimiter='\t')
    
            # save and create plots (shifts)
#            analysis.stat_utils().save_plots(df=endTidal, O2_time=pre_O2.Time, O2=pre_O2.Data, O2_shift=processed_O2, O2_correlation=O2_corr, O2_shift_f=processed_O2,
#                                             CO2_time=pre_CO2.Time, CO2=pre_CO2.Data, CO2_shift=processed_CO2, CO2_correlation=CO2_corr, CO2_shift_f=processed_CO2, meants=meants,
#                                             coeff=coeff, f_path=f_path, key=key, verb=verb, time_points=time_pts, TR=tr)
            analysis.stat_utils().save_plots(df=endTidal, O2_time=pre_O2.Time, O2=pre_O2.Data, O2_shift=processed_O2, O2_correlation=O2_corr, O2_shift_f=processed_O2_f,
                                             CO2_time=pre_CO2.Time, CO2=pre_CO2.Data, CO2_shift=processed_CO2, CO2_correlation=CO2_corr, CO2_shift_f=processed_CO2_f, meants=meants,
                                             coeff=coeff, coeff_f=coeff_f, comb_corr=comb_corr,
                                             f_path=f_path, key=key, verb=verb, time_points=time_pts, TR=tr)
            # analysis.stat_utils().save_plots(df=endTidal, O2=pre_O2, O2_shift=processed_O2, CO2=pre_CO2, CO2_shift=processed_CO2, meants=meants, f_path=graphs_dir+id, key=key, verb=verb, TR=interp_time)

    
    if verb:
        print()
            
    if verb:
        print('Finished processing each patient')
    
    if verb:
        print()
    
    #construct new DataFrame
    et_frame = pd.DataFrame(ET_dict)
    
    #merge and drop bad entries
    df = p_df.merge(et_frame, on=['Cohort', 'ID', 'Date'], how='outer')
    df = df[df.ET_exists != False].drop('ET_exists', axis = 1)
    
    #reset indeces
    df = df.reset_index(drop=True)
    
    #    print("df head----------------\n",p_df.head())
    
#    if verb:
#        print('\n\nStarting to run feat')
#    #run Feat
#    #check for (and make) feat directory
#    if not os.path.exists(feat_dir):
#        os.mkdir(feat_dir)
#    
#    #make design file directory
#    if not os.path.exists(feat_dir+'design_files/'):
#        os.mkdir(feat_dir+'design_files/')
#    
#    # load design template
#    with open(feat_dir+'design_files/template', 'r') as template:
#        stringTemp = template.read()
#        for i in range(len(df)):
#            output_dir = feat_dir+key+df.Cohort[i]+df.ID[i]+'_'+df.Date[i]
##            output_dir = '/media/ke/8tb_part2/FSL_work/feat/both_shift/'+key+df.Cohort[i]+df.ID[i]+'_'+df.Date[i]
#            if os.path.exists(output_dir+'.feat'):
#                if verb:
#                    print('FEAT already exists for', key+df.Cohort[i]+df.ID[i]+'_'+df.Date[i])
#                if over:
#                    if verb:
#                        print('Overwriting')
#                    subprocess.run(['rm', '-rf', output_dir+'.feat'])
#                else:
#                    continue
#            to_write = stringTemp[:]
#            # print(to_write)
#            to_write = to_write.replace("%%OUTPUT_DIR%%",'"'+output_dir+'"')
#            to_write = to_write.replace("%%VOLUMES%%",'"'+str(df.Volumes[i])+'"')
#            to_write = to_write.replace("%%TR%%",'"'+str(df.eff_TR[i])+'"')
#            to_write = to_write.replace("%%BOLD_FILE%%",'"'+df.BOLD_path[i]+'"')
#            to_write = to_write.replace("%%FS_T1%%",'"'+df.T1_path[i]+'"')
#            to_write = to_write.replace("%%O2_CONTRAST%%",'"'+df.ETO2[i]+'"')
#            to_write = to_write.replace("%%CO2_CONTRAST%%",'"'+df.ETCO2[i]+'"')
#    
#            ds_path = feat_dir+'design_files/'+key+df.Cohort[i]+df.ID[i]+'_'+df.Date[i]+'.fsf'
#            with open(ds_path, 'w+') as outFile:
#                outFile.write(to_write)
#                        
#            index = analysis.parallel_processing().get_next_avail(processes, verb, limit, key, 'FEAT')
#            
#            if verb:
#                print('Starting FEAT')
#            processes[index] = subprocess.Popen(['feat', ds_path])
#            time.sleep(0.5)
#        
#        analysis.parallel_processing().wait_remaining(processes, verb, key, 'FEAT')
#        
#    # run featquery
#    for i in range(len(df)):
#        p_id = key+df.Cohort[i]+df.ID[i]+'_'+df.Date[i]
#        feat_output_dir = feat_dir+p_id+'.feat/'
##        feat_output_dir = '/media/ke/8tb_part2/FSL_work/feat/both_shift/'+p_id+'.feat/'
#        
#        O2_mask_dir_path = feat_output_dir+'cluster_mask_zstat1.nii.gz'
#        CO2_mask_dir_path = feat_output_dir+'cluster_mask_zstat2.nii.gz'
#                    
#        index = analysis.parallel_processing().get_next_avail(processes, verb, limit, key, 'featquery')
#        
#        if os.path.exists(feat_output_dir+'fq_O2'):
#            if verb:
#                print('O2 featquery already exists for', p_id)
#            if over:
#                if verb:
#                    print('Overwriting')
#                processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope1', 'fq_O2', '-p', '-s', O2_mask_dir_path])
#        else:
#            if verb:
#                print('Starting O2 featquery for', p_id)
#            processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope1', 'fq_O2', '-p', '-s', O2_mask_dir_path])
#        
#        index = analysis.parallel_processing().get_next_avail(processes, verb, limit, key, 'featquery')
#        
#        if os.path.exists(feat_output_dir+'fq_CO2'):
#            if verb:
#                print('CO2 featquery already exists for', p_id)
#            if over:
#                if verb:
#                    print('Overwriting')
#                processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope2', 'fq_CO2', '-p', '-s', CO2_mask_dir_path])
#        else:
#            if verb:
#                print('Starting featquery for CO2')
#            processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope2', 'fq_CO2', '-p', '-s', CO2_mask_dir_path])
#    
#
#    analysis.parallel_processing().wait_remaining(processes, verb, key, 'featquery')
        
    # get the stats
    for i in range(len(df)):        
        add = True
        
#        output_dir = feat_dir+key+df.Cohort[i]+df.ID[i]+'_'+df.Date[i]
##        output_dir = '/media/ke/8tb_part2/FSL_work/feat/both_shift/'+key+df.Cohort[i]+df.ID[i]+'_'+df.Date[i]
#        feat_output_dir = output_dir+'.feat/'
#        
#        try:
#            cz1 = pd.read_csv(feat_output_dir+'cluster_zstat1.txt', sep='\t', usecols=['Voxels', '-log10(P)', 'Z-MAX', 'COPE-MEAN'])
#            t_vol = cz1.Voxels.sum()
#            
#            for j in range(len(cz1)):
#                cz1.iloc[j] = cz1.iloc[j] * cz1.iloc[j].Voxels/t_vol
#            
#
#            z1 = { 'ID' : [df.Cohort[i]+df.ID[i]+'_'+df.Date[i]],
#                   'type' : [key],
#                   'Voxels': [t_vol],
#                   '-log10(p)' : [cz1['-log10(P)'].sum()],
#                   'COPE-MEAN' : [cz1['COPE-MEAN'].sum()]}
#            cz1_final = pd.DataFrame(z1)
#        
#        except FileNotFoundError:
#            warnings['ID'].append(df.Cohort[i] + df.ID[i] + '_' + df.Date[i])
#            warnings['warning'].append('No cluster_zstat1.txt')
#            add = False
#            if verb:
#                print('No cluster_zstat1.txt')
#        
#        try:
#            cz2 = pd.read_csv(feat_output_dir+'cluster_zstat2.txt', sep='\t', usecols=['Voxels', '-log10(P)', 'Z-MAX', 'COPE-MEAN'])
#            t_vol = cz2.Voxels.sum()
#            
#            for j in range(len(cz2)):
#                cz2.iloc[j] = cz2.iloc[j] * cz2.iloc[j].Voxels/t_vol
#            
#
#            z2 = { 'ID' : [df.Cohort[i]+df.ID[i]+'_'+df.Date[i]],
#                   'type' : [key],
#                   'Voxels': [t_vol],
#                   '-log10(p)' : [cz2['-log10(P)'].sum()],
#                   'COPE-MEAN' : [cz2['COPE-MEAN'].sum()]}
#            cz2_final = pd.DataFrame(z2)
#        
#        except FileNotFoundError:
#            warnings['ID'].append(df.Cohort[i] + df.ID[i] + '_' + df.Date[i])
#            warnings['warning'].append('No cluster_zstat2.txt')
#            add = False
#            if verb:
#                print('No cluster_zstat2.txt', df.ID[i], '_', df.Date[i])
#        
#        build = cz1_final.merge(cz2_final, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
#        
#        O2_mask_dir_path = feat_output_dir+'cluster_mask_zstat1.nii.gz'
#        CO2_mask_dir_path = feat_output_dir+'cluster_mask_zstat2.nii.gz'
#            
#        O2 = feat_output_dir+'fq_O2/'
#        try:
#            fq1 = pd.read_csv(O2+'report.txt', sep='\t| ', header=None, usecols=[5], engine='python')
#            fq1 = fq1.rename(columns={5 : 'fq_mean'})
#            fq1['ID'] = df.Cohort[i]+df.ID[i]+'_'+df.Date[i]
#            fq1['type'] = key
#            fq1 = fq1[['ID', 'type', 'fq_mean']]
#            build = build.merge(fq1, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
#        except FileNotFoundError:
#            warnings['ID'].append(df.Cohort[i] + df.ID[i] + '_' + df.Date[i])
#            warnings['warning'].append('No O2 activation found')
#            add = False
#            if verb:
#                print('No O2 activation found for', df.ID[i] + '_' + df.Date[i], 'O2')
#        
#            
#        CO2 = feat_output_dir+'fq_CO2/'
#        try:
#            fq2 = pd.read_csv(CO2+'report.txt', sep='\t| ', header=None, usecols=[5], engine='python')
#            fq2 = fq2.rename(columns={5 : 'fq_mean'})
#            fq2['ID'] = df.Cohort[i]+df.ID[i]+'_'+df.Date[i]
#            fq2['type'] = key
#            fq2 = fq2[['ID', 'type', 'fq_mean']]
#            build = build.merge(fq2, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
#        except FileNotFoundError:
#            warnings['ID'].append(df.Cohort[i] + df.ID[i] + '_' + df.Date[i])
#            warnings['warning'].append('No CO2 activation found')
#            add = False
#            if verb:
#                print('No CO2 activation found for', df.ID[i] + '_' + df.Date[i], 'O2')
#        
#        build['O2_shift'] = df.O2_shift[i]
#        build['CO2_shift'] = df.CO2_shift[i]
#        build['O2_coeff'] = df.coeffs[i][0]
#        build['CO2_coeff'] = df.coeffs[i][1]
#        build['r'] = df.r[i]
#        build['p_value'] = df.p_value[i]
        
        build = pd.DataFrame({'ID' : [df.Cohort[i]+df.ID[i]+'_'+df.Date[i]],
                              'O2_shift' : [df.O2_shift[i]],
                              'CO2_shift' : [df.CO2_shift[i]],
                              'O2_coeff' : [df.coeffs[i][0]],
                              'CO2_coeff' : [df.coeffs[i][1]],
                              'r' : [df.r[i]],
                              'p_value' : [df.p_value[i]]})
        
        if add:
            stats_df = pd.concat([stats_df, build])
    
    if verb:
        print()
    
    stats_df.reset_index(drop=True)

warnings_df = pd.DataFrame(warnings).sort_values('ID')
stats_df = stats_df.sort_values('ID')

with pd.ExcelWriter(path+'stats_data_comb.xlsx') as writer:  # doctest: +SKIP
    stats_df.to_excel(writer, sheet_name='Stats', index=False)
    warnings_df.to_excel(writer, sheet_name='Warnings', index=False)

if verb:
    print('============== Script Finished ==============')


