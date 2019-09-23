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
parser.add_argument("-w", "--wh", action='store_true', help='process WH cohort')


#get the positional arguments
args = parser.parse_args()
path = args.path

verb = True if args.verbose else False
four = True if args.fouier else False
trough = True if args.CO2_trough else False
over = True if args.overwrite else False
block = True if args.block else False
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
    
feat_dir = '/media/ke/8tb_part2/FSL_work/feat_test/'
    

# set the limit for the number of processes (10 less that the total number of cores in the system) that can be run at once
cores = multiprocessing.cpu_count()
limit = cores - 5 if cores > 8 else 1
processes = [None] * limit

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
    
    fs_files = [file for file in os.listdir(freesurfer_t1_dir) if file == p_df.Cohort[i]+p_df.ID[i]+'_FS_T1.nii.gz']

    #select and add file to appropriate list
    b_temp = patient_dir +'/BOLD/'+b_files[0] if len(b_files) > 0 else ''
    t_temp = freesurfer_t1_dir+fs_files[0] if len(fs_files) > 0 else ''

    nii_paths['BOLD_path'].append(b_temp)
    nii_paths['T1_path'].append(t_temp)
    nii_paths['boldFS_exists'].append(len(b_files) > 0 and len(fs_files)>0)
    
    if len(fs_files) == 0:
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
#choose non-trivial bold series
warning = p_df[p_df.Volumes < 150]
for i in range(len(warning)):
    print('\t\t' + warning[i].Cohort + warning[i].ID + '_' + warning[i].Date + 'has a total volume < 150')
#print('\np_df w/ dim > 150\n',p_df.head())

####run EndTidal Cleaning and return paths

stats_df = pd.DataFrame()

for typ in ['four', 'peak', 'trough']:
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

    #generate cleaned data header
    if four:
        key = 'f_'
    #        key = sm+'f_'
    elif trough:
        key = 't_'
    #        key = sm+'t_'
    elif block:
        key = 'b_'
    #        key = sm+'b_'
    else:
        key = 'p_'
    #        key = sm+'p_'
    
    ET_dict = {'ETO2' : [], 'ETCO2' : [], 'ET_exists' : [], 'Cohort' : [], 'ID' : [], 'Date' : []}
    
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
    
        #check if the save_files already exist
        if(verb):
            print('\tID: ', cohort + id + '_' + date," \t Grabbing atlas")
        ET_dict['ETO2'].append(save_O2)
        ET_dict['ETCO2'].append(save_CO2)
        ET_dict['ET_exists'].append(True)
        processed_O2 = np.loadtxt(save_O2)
        processed_O2 = signal.resample(processed_O2, len(meants))
        processed_CO2 = np.loadtxt(save_CO2)
        processed_CO2 = signal.resample(processed_CO2, len(meants))
    
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
    with open(feat_dir+'design_files/template', 'r') as template:
        stringTemp = template.read()
        for i in range(len(df)):
            output_dir = feat_dir+key+df.Cohort[i]+df.ID[i]+'_'+df.Date[i]
            if os.path.exists(output_dir+'.feat'):
                if verb:
                    print('FEAT already exists for', key+df.Cohort[i]+df.ID[i]+'_'+df.Date[i])
                if over:
                    if verb:
                        print('Overwriting')
                    subprocess.Popen(['rm', '-rf', output_dir+'.feat'])
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
    
            ds_path = feat_dir+'design_files/'+key+df.Cohort[i]+df.ID[i]+'_'+df.Date[i]+'.fsf'
            with open(ds_path, 'w+') as outFile:
                outFile.write(to_write)
                        
            index = analysis.parallel_processing().get_next_avail(processes, verb, limit, key, 'FEAT')
            
            if verb:
                print('Starting FEAT')
            processes[index] = subprocess.Popen(['feat', ds_path])
            time.sleep(0.5)
        
        analysis.parallel_processing().wait_remaining(processes, verb, key, 'FEAT')
        
        # run featquery
        for i in range(len(p_df)):
            p_id = key+df.Cohort[i]+df.ID[i]+'_'+df.Date[i]
            feat_output_dir = feat_dir+p_id+'.feat/'
            
            O2_mask_dir_path = feat_output_dir+'cluster_mask_zstat1.nii.gz'
            CO2_mask_dir_path = feat_output_dir+'cluster_mask_zstat2.nii.gz'
                        
            index = analysis.parallel_processing().get_next_avail(processes, verb, limit, key, 'featquery')
            
            if os.path.exists(feat_output_dir+'fq_O2'):
                if verb:
                    print('O2 featquery already exists for', p_id)
                if over:
                    if verb:
                        print('Overwriting')
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
                    processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope2', 'fq_CO2', '-p', '-s', CO2_mask_dir_path])
            else:
                if verb:
                    print('Starting featquery for CO2')
                processes[index] = subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope2', 'fq_CO2', '-p', '-s', CO2_mask_dir_path])
        
    
        analysis.parallel_processing().wait_remaining(processes, verb, key, 'featquery')
        
        # get the stats
        for i in range(len(p_df)):        
            output_dir = feat_dir+key+df.Cohort[i]+df.ID[i]+'_'+df.Date[i]
            feat_output_dir = output_dir+'.feat/'
            
            try:
                cz1 = pd.read_csv(feat_output_dir+'cluster_zstat1.txt', sep='\t', usecols=['Voxels', '-log10(P)', 'Z-MAX', 'COPE-MEAN'])
                t_vol = cz1.Voxels.sum()
                
                for j in range(len(cz1)):
                    cz1.iloc[j] = cz1.iloc[j] * cz1.iloc[j].Voxels/t_vol
                
    
                z1 = { 'ID' : [df.Cohort[i]+df.ID[i]+'_'+df.Date[i]],
                       'type' : [key],
                       '-log10(p)' : [cz1['-log10(P)'].sum()],
                       'Z-MAX' : [cz1['Z-MAX'].sum()],
                       'COPE-MEAN' : [cz1['COPE-MEAN'].sum()]}
                cz1_final = pd.DataFrame(z1)
            
            except FileNotFoundError:
                if verb:
                    print('No O2 activation found')
                continue
            
            try:
                cz2 = pd.read_csv(feat_output_dir+'cluster_zstat2.txt', sep='\t', usecols=['Voxels', '-log10(P)', 'Z-MAX', 'COPE-MEAN'])
                t_vol = cz2.Voxels.sum()
                
                for j in range(len(cz2)):
                    cz2.iloc[j] = cz2.iloc[j] * cz2.iloc[j].Voxels/t_vol
                
    
                z2 = { 'ID' : [df.Cohort[i]+df.ID[i]+'_'+df.Date[i]],
                       'type' : [key],
                       '-log10(p)' : [cz2['-log10(P)'].sum()],
                       'Z-MAX' : [cz2['Z-MAX'].sum()],
                       'COPE-MEAN' : [cz2['COPE-MEAN'].sum()]}
                cz2_final = pd.DataFrame(z2)
            
            except FileNotFoundError:
                if verb:
                    print('No CO2 activation found')
                continue
            
            build = cz1_final.merge(cz2_final, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
            
            O2_mask_dir_path = feat_output_dir+'cluster_mask_zstat1.nii.gz'
            CO2_mask_dir_path = feat_output_dir+'cluster_mask_zstat2.nii.gz'
                
            O2 = feat_output_dir+'fq_O2/'
            try:
                fq1 = pd.read_csv(O2+'report.txt', sep='\t| ', header=None, usecols=[5], engine='python')
                fq1 = fq1.rename(columns={5 : 'mean'})
                fq1['ID'] = df.Cohort[i]+df.ID[i]+'_'+df.Date[i]
                fq1['type'] = key
                fq1 = fq1[['ID', 'type', 'mean']]
                build = build.merge(fq1, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
            except FileNotFoundError:
                if verb:
                    print('No O2 activation found')
                continue
            
                
            CO2 = feat_output_dir+'fq_CO2/'
            try:
                fq2 = pd.read_csv(CO2+'report.txt', sep='\t| ', header=None, usecols=[5], engine='python')
                fq2 = fq2.rename(columns={5 : 'mean'})
                fq2['ID'] = df.Cohort[i]+df.ID[i]+'_'+df.Date[i]
                fq2['type'] = key
                fq2 = fq2[['ID', 'type', 'mean']]
                build = build.merge(fq2, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
            except FileNotFoundError:
                if verb:
                    print('No CO2 activation found')
                continue
            
            stats_df = pd.concat([stats_df, build])
        
        if verb:
            print()
    
    stats_df.reset_index(drop=True)


stats_df.to_excel(path+'stats_data.xlsx', index=False)

if verb:
    print('============== Script Finished ==============')