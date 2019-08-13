'''
By: Austin Sophonsri, Jimi Cao

'''

import os
import subprocess
import argparse
import pandas as pd
import numpy as np
import analysis
import json

# instantiate the argument parser
parser = argparse.ArgumentParser()

# add positional argument
parser.add_argument("path", help="path of the folder that contains all the endtidal edit files")

# add optional arguments
parser.add_argument("-v", "--verbose", action='store_true', help="incrase output verbosity")
parser.add_argument("-f", "--fouier", action='store_true', help='switch analysis to fouier instead of default peak_find')

#get the positional arguments
args = parser.parse_args()
path = args.path

verb = (True if args.verbose else False)
four = True if args.fouier else False

###########set directories (TODO, automate)
nifti_dir = '/home/ke/Desktop/FSL_work/SH_info/'
freesurfer_t1_dir = '/home/ke/Desktop/FSL_work/SH_FST1/'
feat_dir = '/home/ke/Desktop/FSL_work/feat/'


# make sure the path ends with '/'
if path[-1] != '/':
    path += '/'

# all grab all the .txt files in the endtidal folder
txt_files = [file for file in os.listdir(path) if file.endswith('edits.txt')]

#separate patient ID and scan date and pack into tuple
p_df = pd.DataFrame({
                'Cohort':[f[0:2] for f in txt_files],
                'ID':[f[3:6] for f in txt_files],
                'Month':[f[11:13] for f in txt_files],
                'Day':[f[13:15] for f in txt_files],
                'Year':[f[7:11] for f in txt_files],
                'EndTidal_Path': [path+f for f in txt_files]
             })

#print(p_df.head())
#create patient bold scan listdir (is a list not DataFrame)
patient_BOLDS_header = [p_df.Cohort[i]+p_df.ID[i]+'_BOLD_'+p_df.Year[i]+p_df.Month[i]+p_df.Day[i]
                    for i in range(len(p_df))]


#get bold and FS paths
nii_paths = {'BOLD_path' : [], 'T1_path' : [], 'boldFS_exists': []}
for i in range(len(p_df)):
    #if bold file doesnt exist then continue
    if(not os.path.exists(nifti_dir + p_df.Cohort[i] + p_df.ID[i] +'/BOLD')):
        nii_paths['BOLD_path'].append('')
        nii_paths['T1_path'].append('')
        nii_paths['boldFS_exists'].append(False)
        continue

    #get all matching files
    b_files = [file for file in os.listdir(nifti_dir + p_df.Cohort[i] + p_df.ID[i] +'/BOLD/') if file.endswith('a.nii')]
    fs_files = [file for file in os.listdir(freesurfer_t1_dir) if file == p_df.Cohort[i]+p_df.ID[i]+'_FS_T1.nii.gz']

    #select and add file to appropriate list
    nii_paths['BOLD_path'].append(nifti_dir + p_df.Cohort[i] + p_df.ID[i] +'/BOLD/'+b_files[0] if len(b_files) > 0 else '')
    nii_paths['T1_path'].append(freesurfer_t1_dir+fs_files[0] if len(fs_files) > 0 else '')
    nii_paths['boldFS_exists'].append(len(b_files) > 0 and len(fs_files)>0)


#append bold, FS paths, and conditional to p_df
p_df = pd.concat((p_df, pd.DataFrame(nii_paths)), axis=1)

#drop all false conditional rows and conditional column and reset indeces
p_df = p_df[p_df.boldFS_exists != False].drop('boldFS_exists', axis = 1)
p_df = p_df.reset_index(drop=True)

#get json files and read CSV
tr_dict = {'TR' : []}
for f_path in p_df.BOLD_path:
    #print(f_path[:-5]+'.json')
    with open(f_path[:-5]+'.json', 'r') as j_file:
        data = json.load(j_file)
        #print(data['RepetitionTime'])
        tr_dict['TR'].append(data['RepetitionTime'])

p_df = pd.concat((p_df, pd.DataFrame(tr_dict)), axis=1)
#print('\n',p_df.head())

#get number of volumes
#run fslinfo and pipe -> python buffer -> replace \n with ' ' -> split into list -> choose correct index -> convert string to int
p_df["Dimension"] = [int(subprocess.run(['fslinfo' ,b_path], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n',' ').split()[9]) for b_path in p_df.BOLD_path]
#choose non-trivial bold series
p_df = p_df[p_df.Dimension >=150]
p_df = p_df.reset_index(drop=True)
#print('\np_df w/ dim > 150\n',p_df.head())

####run EndTidal Cleaning and return paths
#make endtidal folders

ET_dict = {'ETO2' : [], 'ETCO2' : [], 'ET_exists' : [], 'ID2' : []}

for f_path, dim, id in zip(p_df.EndTidal_Path, p_df.Dimension, p_df.ID):
    if not os.path.exists(f_path[:-4]):
        os.mkdir(f_path[:-4])

    #perfrom cleaning and load into p_df
    #load data into DataFrame
    ET_dict['ID2'].append(id)
    endTidal = pd.read_csv(f_path, sep='\t|,', names=['Time', 'O2', 'CO2', 'thrw', 'away'], usecols=['Time', 'O2', 'CO2'], index_col=False, engine='python')
    #drop rows with missing cols
    endTidal = endTidal.dropna()

    #skip if DataFrame is empty
    if endTidal.empty:
        os.rmdir(f_path[:-4])
        ET_dict['ETO2'].append('')
        ET_dict['ETCO2'].append('')
        ET_dict['ET_exists'].append(False)
        continue

    # need to scale CO2 data is necessary
    if endTidal.CO2.max() < 1:
        endTidal.CO2 = endTidal.CO2 * 100
        
    interp_time = 510.0/dim

    if four:
        #get fourier cleaned data
        processed_O2 = analysis.fourier_filter(endTidal.Time, endTidal.O2, 3, 35, interp_time)
        processed_CO2 = analysis.fourier_filter(endTidal.Time, endTidal.CO2, 3, 35, interp_time)
    else:
        #get peak data
        processed_CO2, processed_O2 = analysis.get_peaks(df=endTidal, verb=verb, file=f_path, TR=interp_time)
    

    #generate cleaned data paths
    save_O2 = f_path[:-4]+'/O2_contrast.txt'
    save_CO2 = f_path[:-4]+'/CO2_contrast.txt'

    #storing cleaned data paths
    ET_dict['ETO2'].append(save_O2)
    ET_dict['ETCO2'].append(save_CO2)
    ET_dict['ET_exists'].append(True)

    #save data
    np.savetxt(save_O2, processed_O2, delimiter='\t')
    np.savetxt(save_CO2, processed_CO2, delimiter='\t')
    
    # save and create plots
    analysis.save_plots(df=endTidal, O2=processed_O2, CO2=processed_CO2, f_path=f_path, verb=verb, TR=interp_time)

#construct new DataFrame
et_frame = pd.DataFrame(ET_dict)

#concat and rop bad dataframes
p_df = pd.concat((p_df, pd.DataFrame(ET_dict)), axis=1)
p_df = p_df[p_df.ET_exists != False].drop('ET_exists', axis = 1)

#reset indeces
p_df = p_df.reset_index(drop=True)

print(p_df.head())

#run Feat
#check for (and make) feat directory
if not os.path.exists(feat_dir):
    os.mkdir(feat_dir)

#make design file directory
if not os.path.exists(feat_dir+'design_files/'):
    os.mkdir(feat_dir+'design_files/')

#load design template
with open(feat_dir+'design_files/template', 'r') as template:
    stringTemp = template.read()
    for i in range(len(p_df)):
        output_dir = feat_dir+p_df.Cohort[i]+p_df.ID[i]
        # if not os.path.exists(output_dir):
        #     os.mkdir(output_dir)
        to_write = stringTemp[:]
        # print(to_write)
        to_write = to_write.replace("%%OUTPUT_DIR%%",'"'+output_dir+'"')
        to_write = to_write.replace("%%VOLUMES%%",'"'+str(p_df.Dimension[i])+'"')
        to_write = to_write.replace("%%TR%%",'"'+str(p_df.TR[i])+'"')
        to_write = to_write.replace("%%BOLD_FILE%%",'"'+p_df.BOLD_path[i]+'"')
        to_write = to_write.replace("%%FS_T1%%",'"'+p_df.T1_path[i]+'"')
        to_write = to_write.replace("%%O2_CONTRAST%%",'"'+p_df.ETO2[i]+'"')
        to_write = to_write.replace("%%CO2_CONTRAST%%",'"'+p_df.ETCO2[i]+'"')

        with open(feat_dir+'design_files/'+p_df.ID[i]+'.fsf', 'w+') as outFile:
            outFile.write(to_write)
            
        os.spawnlp(os.P_NOWAIT, 'feat', 'feat', feat_dir+'design_files/'+p_df.ID[i]+'.fsf')
        # print('written')








#
