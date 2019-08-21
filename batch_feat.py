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
import matplotlib.pyplot as  plt
from sklearn import metrics

# instantiate the argument parser
parser = argparse.ArgumentParser()

# add positional argument
parser.add_argument("path", help="path of the folder that contains all the endtidal edit files")

# add optional arguments
parser.add_argument("-v", "--verbose", action='store_true', help="incrase output verbosity")
parser.add_argument("-f", "--fouier", action='store_true', help='switch analysis to fouier instead of default peak_find')
parser.add_argument("-t", "--CO2_trough", action='store_true', help='switch CO2 peak finding to troughs')
parser.add_argument("-o", "--overwrite", action='store_true', help='overwrite existing processed gas data')


#get the positional arguments
args = parser.parse_args()
path = args.path

verb = (True if args.verbose else False)
four = True if args.fouier else False
trough = True if args.CO2_trough else False
over = False if args.overwrite else True

###########set directories (TODO, automate)
home_dir = '/media/ke/8tb_part2/FSL_work/'
nifti_dir = '/media/ke/8tb_part2/FSL_work/SH_info/'
processed_dir = '/media/ke/8tb_part2/FSL_work/SH_info/BOLD_processed'
freesurfer_t1_dir = '/media/ke/8tb_part2/FSL_work/SH_FST1/'
feat_dir = '/media/ke/8tb_part2/FSL_work/feat/'


# make sure the path ends with '/'
if path[-1] != '/':
    path += '/'

# all grab all the .txt files in the endtidal folder
txt_files = [file for file in os.listdir(path) if file.endswith('edits.txt')]

if verb:
    print("Constructing dataframe for patients")
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

if verb:
    print('Constructing dataframe that holds all relavent paths')
#get bold and FS paths
nii_paths = {'BOLD_path' : [], 'BOLD_corrected_path': [], 'T1_path' : [], 'meants_path': [], 'Processed_path': [], 'boldFS_exists': []}
for i in range(len(p_df)):
    if verb:
        print('\tGetting paths relavent to',p_df.Cohort[i] + p_df.ID[i])
    #if bold file doesnt exist then continue
    if(not os.path.exists(nifti_dir + p_df.Cohort[i] + p_df.ID[i] +'/BOLD')):
        nii_paths['BOLD_path'].append('')
        nii_paths['BOLD_corrected_path'].append('')
        nii_paths['T1_path'].append('')
        nii_paths['meants_path'].append('')
        nii_paths['Processed_path'].append('')
        nii_paths['boldFS_exists'].append(False)

    else:
        #get all matching files
        b_files = [file for file in os.listdir(nifti_dir + p_df.Cohort[i] + p_df.ID[i] +'/BOLD/') if file.endswith('.nii')]
        fs_files = [file for file in os.listdir(freesurfer_t1_dir) if file == p_df.Cohort[i]+p_df.ID[i]+'_FS_T1.nii.gz']

        #select and add file to appropriate list
        b_temp = nifti_dir + p_df.Cohort[i] + p_df.ID[i] +'/BOLD/'+b_files[0] if len(b_files) > 0 else ''
        t_temp = freesurfer_t1_dir+fs_files[0] if len(fs_files) > 0 else ''

        nii_paths['BOLD_path'].append(b_temp)
        nii_paths['T1_path'].append(t_temp)
        nii_paths['boldFS_exists'].append(len(b_files) > 0 and len(fs_files)>0)

        if(len(b_files) > 0):
            #construct the processed nifti directory
            processed_dir = nifti_dir + p_df.Cohort[i] + p_df.ID[i] +'/BOLD_processed/'
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
            print('\tAll relavent files grabbed')

if verb:
    print('Concatenating patient dataframe with path dataframe')
#append bold, FS paths, and conditional to p_df
p_df = pd.concat((p_df, pd.DataFrame(nii_paths)), axis=1)

if verb:
    print('Dropping patients that have missing data')

#drop all false conditional rows and conditional column and reset indeces
p_df = p_df[p_df.boldFS_exists != False].drop('boldFS_exists', axis = 1)
p_df = p_df.reset_index(drop=True)


if verb:
    print('Getting the TR for each patient')
#get json files and read CSV
tr_dict = {'TR' : []}

for b_path in p_df.BOLD_path:
    #print(f_path[:-5]+'.json')
    with open(b_path[:-4]+'.json', 'r') as j_file:
        data = json.load(j_file)
        #print(data['RepetitionTime'])
        tr_dict['TR'].append(data['RepetitionTime'])

p_df = pd.concat((p_df, pd.DataFrame(tr_dict)), axis=1)
#print('\n',p_df.head())

#get number of volumes
#run fslinfo and pipe -> python buffer -> replace \n with ' ' -> split into list -> choose correct index -> convert string to int
if verb:
    print('Getting the dimension of the BOLD for each patient with fslinfo')
p_df["Dimension"] = [int(subprocess.run(['fslinfo' ,b_path], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n',' ').split()[9]) for b_path in p_df.BOLD_path]

if verb:
    print('Dropping patients with a dimension < 150')
#choose non-trivial bold series
p_df = p_df[p_df.Dimension >=150]
p_df = p_df.reset_index(drop=True)
#print('\np_df w/ dim > 150\n',p_df.head())

####run EndTidal Cleaning and return paths

#r2

for type in ['four', 'peak', 'trough']:
    if type == 'four':
        four = True
        trough = False
    elif type == 'trough':
        four = False
        trough = True
    else:
        four = False
        trough = False

    #generate cleaned data header
    key = ''
    if four:
        key = 'f_'
    elif trough:
        key = 't_'
    else:
        key = 'p_'

    ET_dict = {'ETO2' : [], 'ETCO2' : [], 'ET_exists' : [], 'ID2' : []}

    if verb:
        print('Start processing each patient')
    for f_path, dim, id, b_path, p_path, meants_path in zip(p_df.EndTidal_Path, p_df.Dimension, p_df.ID, p_df.BOLD_corrected_path, p_df.Processed_path, p_df.meants_path):
        if not os.path.exists(f_path[:-4]):
            os.mkdir(f_path[:-4])

        ##path made and stored in p_df.Processed_path
        ##pre-running slicetimer and mcflirt (motion correction) as we create bold paths

        #perfrom cleaning and load into p_df
        #load data into DataFrame
        ET_dict['ID2'].append(id)
        endTidal = pd.read_csv(f_path, sep='\t|,', names=['Time', 'O2', 'CO2', 'thrw', 'away'], usecols=['Time', 'O2', 'CO2'], index_col=False, engine='python')
        #drop rows with missing cols
        endTidal = endTidal.dropna()

        meants = np.loadtxt(meants_path, delimiter='\n')[3:]

        #skip if DataFrame is empty
        if endTidal.empty:
            os.rmdir(f_path[:-4])
            ET_dict['ETO2'].append('')
            ET_dict['ETCO2'].append('')
            ET_dict['ET_exists'].append(False)
            print("patient: ", id, " is has empty end-tidal")
            continue

        #generate cleaned data paths
        save_O2 = f_path[:-4]+'/'+key+'O2_contrast.txt'
        save_CO2 = f_path[:-4]+'/'+key+'CO2_contrast.txt'

        #check if the save_files already exist
        if(os.path.exists(save_O2) and os.path.exists(save_CO2) and over):
            if(verb):
                print('\tID: ',id," \tProcessed gas files already exist")
            ET_dict['ETO2'].append(save_O2)
            ET_dict['ETCO2'].append(save_CO2)
            ET_dict['ET_exists'].append(True)
            continue

        # need to scale CO2 data is necessary
        if endTidal.CO2.max() < 1:
            endTidal.CO2 = endTidal.CO2 * 100

        interp_time = 480/(dim-3)

        if four:
            if verb:
                print('Starting fourier for', id)
            #get fourier cleaned data
            pre_O2 = analysis.fft_analysis().fourier_filter(endTidal.Time, endTidal.O2, 3, 35, interp_time)
            pre_CO2 = analysis.fft_analysis().fourier_filter(endTidal.Time, endTidal.CO2, 3, 35, interp_time)
            processed_O2 = analysis.shifter().corr_align(meants, pre_O2)
            processed_CO2 = analysis.shifter().corr_align(meants, pre_CO2)
        else:
            #get peak data
            pre_CO2, pre_O2 = analysis.peak_analysis.get_peaks(endTidal, len(meants), verb, f_path, interp_time, trough)

            # get shifted O2 and CO2
            processed_CO2 = analysis.shifter().corr_align(meants, pre_CO2)
            processed_O2 = analysis.shifter().corr_align(meants, pre_O2)


        #fit to linear model
        coeffs = analysis.optimizer().stochastic_optimize_GLM(processed_O2, processed_CO2, meants, lifespan=1000)

        #generate prediction
        peak_prediction = coeffs[0]*processed_O2 + coeffs[1]*processed_CO2 +coeffs[2]

        #get r^2
        print("Regression score for:",id,' is ', metrics.r2_score(meants, peak_prediction))

        #generate cleaned data paths
        save_O2 = f_path[:-4]+'/'+key+'O2_contrast.txt'
        save_CO2 = f_path[:-4]+'/'+key+'CO2_contrast.txt'

        #storing cleaned data paths
        ET_dict['ETO2'].append(save_O2)
        ET_dict['ETCO2'].append(save_CO2)
        ET_dict['ET_exists'].append(True)

        #save data
        np.savetxt(save_O2, processed_O2, delimiter='\t')
        np.savetxt(save_CO2, processed_CO2, delimiter='\t')

        # save and create plots
        analysis.stat_utils().save_plots(df=endTidal, O2=pre_O2, O2_shift=processed_O2, CO2=pre_CO2, CO2_shift=processed_CO2, meants=meants, f_path=f_path, key=key, verb=verb, TR=interp_time)
        plt.figure(figsize=(20,10))
        plt.plot(meants)
        plt.plot(peak_prediction)
        plt.savefig(f_path[:-4]+'/'+key+'regression_plot.png')
        plt.clf()
        plt.close()
    if verb:
        print('Finished processing each patient')
    #construct new DataFrame
    et_frame = pd.DataFrame(ET_dict)

    #concat and rop bad dataframes
    p_df = pd.concat((p_df, pd.DataFrame(ET_dict)), axis=1)
    p_df = p_df[p_df.ET_exists != False].drop('ET_exists', axis = 1)

    #reset indeces
    p_df = p_df.reset_index(drop=True)

    #print(p_df.head())

    # if verb:
    #     print('Starting to run feat')
    # #run Feat
    # #check for (and make) feat directory
    # if not os.path.exists(feat_dir):
    #     os.mkdir(feat_dir)
    #
    # #make design file directory
    # if not os.path.exists(feat_dir+'design_files/'):
    #     os.mkdir(feat_dir+'design_files/')
    #
    # #load design template
    # with open(feat_dir+'design_files/template', 'r') as template:
    #     stringTemp = template.read()
    #     for i in range(len(p_df)):
    #         output_dir = feat_dir+p_df.Cohort[i]+p_df.ID[i]
    #         # if not os.path.exists(output_dir):
    #         #     os.mkdir(output_dir)
    #         to_write = stringTemp[:]
    #         # print(to_write)
    #         to_write = to_write.replace("%%OUTPUT_DIR%%",'"'+output_dir+'"')
    #         to_write = to_write.replace("%%VOLUMES%%",'"'+str(p_df.Dimension[i])+'"')
    #         to_write = to_write.replace("%%TR%%",'"'+str(p_df.TR[i])+'"')
    #         to_write = to_write.replace("%%BOLD_FILE%%",'"'+p_df.BOLD_path[i]+'"')
    #         to_write = to_write.replace("%%FS_T1%%",'"'+p_df.T1_path[i]+'"')
    #         to_write = to_write.replace("%%O2_CONTRAST%%",'"'+p_df.ETO2[i]+'"')
    #         to_write = to_write.replace("%%CO2_CONTRAST%%",'"'+p_df.ETCO2[i]+'"')
    #
    #         with open(feat_dir+'design_files/'+p_df.ID[i]+'.fsf', 'w+') as outFile:
    #             outFile.write(to_write)
    #
    #         os.spawnlp(os.P_NOWAIT, 'feat', 'feat', feat_dir+'design_files/'+key+p_df.ID[i]+'.fsf', '&')
            # print('written')


if verb:
    print('============== Script Finished ==============')





#
