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
from scipy import signal
import sys
import time

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
parser.add_argument("-r", "--no_r2", action='store_true', help='do not calculate r2 score')


#get the positional arguments
args = parser.parse_args()
path = args.path

verb = (True if args.verbose else False)
four = True if args.fouier else False
trough = True if args.CO2_trough else False
over = True if args.overwrite else False
block = True if args.block else False
no_r2 = True if args.no_r2 else False

# limit the number of process that can be ran at the same time
pro = [None] * 10

###########set directories (TODO, automate)
home_dir = '/media/ke/8tb_part2/FSL_work/'
nifti_dir = '/media/ke/8tb_part2/FSL_work/SH_info/'
processed_dir = '/media/ke/8tb_part2/FSL_work/SH_info/BOLD_processed'
freesurfer_t1_dir = '/media/ke/8tb_part2/FSL_work/SH_FST1/'
feat_dir = '/media/ke/8tb_part2/FSL_work/feat/'
# graphs_dir = '/media/ke/8tb_part2/FSL_work/graphs/'


# make sure the path ends with '/'
if path[-1] != '/':
    path += '/'

# all grab all the .txt files in the endtidal folder
txt_files = [file for file in os.listdir(path) if file.endswith('.txt')]

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
    print('Constructing dataframe that holds all relevant paths')
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
        if verb:
            print('\t\tNo corresponding BOLD folder')

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

r2_dic = {'ID' : [], 'type' : [], 'r2' : []}

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
for typ in ['four', 'peak', 'trough', 'block']:
#    for typ in ['four', 'peak', 'trough']:
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

    ET_dict = {'ETO2' : [], 'ETCO2' : [], 'ET_exists' : [], 'ID2' : []}
    
    to_drop = []

    if verb:
        print('\n\nStart processing each patient with', typ)
    for f_path, dim, cohort, id, b_path, p_path, meants_path in zip(p_df.EndTidal_Path, p_df.Dimension, p_df.Cohort, p_df.ID, p_df.BOLD_corrected_path,
                                                            p_df.Processed_path, p_df.meants_path):
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

        #skip if DataFrame is empty
        if endTidal.empty:
            os.rmdir(f_path[:-4])
            ET_dict['ETO2'].append('')
            ET_dict['ETCO2'].append('')
            ET_dict['ET_exists'].append(False)
            print("\tpatient: ", cohort+id, "has empty end-tidal")
            continue

        meants = np.loadtxt(meants_path, delimiter='\n')
        meants_ts = np.linspace(0,480, len(meants))[3:]
        meants = meants[3:]

        #generate cleaned data paths
        save_O2 = f_path[:-4]+'/'+key+'O2_contrast.txt'
        save_CO2 = f_path[:-4]+'/'+key+'CO2_contrast.txt'

        #check if the save_files already exist
        if(os.path.exists(save_O2) and os.path.exists(save_CO2) and not over):
            if(verb):
                print('\tID: ',cohort+id," \tProcessed gas files already exist")
            ET_dict['ETO2'].append(save_O2)
            ET_dict['ETCO2'].append(save_CO2)
            ET_dict['ET_exists'].append(True)
            processed_O2 = np.loadtxt(save_O2)
            processed_CO2 = np.loadtxt(save_CO2)

        else:
            # need to scale CO2 data is necessary
#            print(endTidal)
#            print()
            
            if endTidal.CO2.max() < 1:
                endTidal.CO2 = endTidal.CO2 * 100

#            print(endTidal)
#            if pre:
            endTidal.CO2 = signal.savgol_filter(endTidal.CO2, 35, 3)
            
            interp_time = 480/(dim-3)

            if four:
                if verb:
                    print('Starting fourier for', cohort+id)
                #get fourier cleaned data
                pre_O2 = analysis.fft_analysis().fourier_filter(endTidal.Time, endTidal.O2, 3, 35, interp_time)
                pre_CO2 = analysis.fft_analysis().fourier_filter(endTidal.Time, endTidal.CO2, 3, 35, interp_time)
            elif block:
                if verb:
                    print('Starting block for', cohort+id)
                pre_O2 = analysis.peak_analysis().block_signal(endTidal.Time, endTidal.O2.apply(lambda x:x*-1), interp_time)*-1
                pre_CO2 = analysis.peak_analysis().block_signal(endTidal.Time, endTidal.CO2, interp_time)

            elif trough:
                if verb:
                    print('Starting troughs for', cohort+id)
                pre_CO2, pre_O2 = analysis.peak_analysis().peak_four(endTidal, verb, f_path, interp_time, True)
            
            else:
                if verb:
                    print('Starting peaks for', cohort+id)
                pre_CO2, pre_O2 = analysis.peak_analysis().peak_four(endTidal, verb, f_path, interp_time)

            # get shifted O2 and CO2
            processed_O2 = analysis.shifter().corr_align(meants, pre_O2)
            processed_CO2 = analysis.shifter().corr_align(meants, pre_CO2)

            #storing cleaned data paths
            ET_dict['ETO2'].append(save_O2)
            ET_dict['ETCO2'].append(save_CO2)
            ET_dict['ET_exists'].append(True)

            #save data
            np.savetxt(save_O2, processed_O2, delimiter='\t')
            np.savetxt(save_CO2, processed_CO2, delimiter='\t')

            # save and create plots (shifts)
            analysis.stat_utils().save_plots(df=endTidal, O2=pre_O2, O2_shift=processed_O2, CO2=pre_CO2, CO2_shift=processed_CO2, meants=meants, f_path=f_path, key=key, verb=verb, TR=interp_time)
            # analysis.stat_utils().save_plots(df=endTidal, O2=pre_O2, O2_shift=processed_O2, CO2=pre_CO2, CO2_shift=processed_CO2, meants=meants, f_path=graphs_dir+id, key=key, verb=verb, TR=interp_time)

        #fit to linear model
        if not no_r2:
            if verb:
                print('Constructing a predicted meants using gas files')
            coeffs = analysis.optimizer().stochastic_optimize_GLM(processed_O2, processed_CO2, meants, lifespan=10000)

            #generate prediction
            peak_prediction = coeffs[0]*processed_O2 + coeffs[1]*processed_CO2 + coeffs[2]

            #get r^2
            regress_score = metrics.r2_score(meants, peak_prediction)
            print("Regression score for:",cohort+id,' is ', regress_score)
            r2_dic['ID'].append(cohort+id)
            r2_dic['type'].append(key)
            r2_dic['r2'].append(regress_score)

            # save and create plots (regression)
            plt.figure(figsize=(20,10))
            plt.plot(meants, label='Meants')
            plt.plot(peak_prediction, label='Prediction')
            plt.legend()
            plt.savefig(f_path[:-4]+'/'+key+'regression_plot.png')
            plt.clf()
            plt.close()

    if verb:
        print('Finished processing each patient')


    #construct new DataFrame
    et_frame = pd.DataFrame(ET_dict)
    
    print(et_frame)
    
    #concat and rop bad dataframes
    df = pd.concat((p_df, pd.DataFrame(ET_dict)), axis=1)
    df = df[df.ET_exists != False].drop('ET_exists', axis = 1)

    #reset indeces
    df = df.reset_index(drop=True)

    # print("df head----------------\n",p_df.head())

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
        skip = False
        for i in range(len(df)):
            output_dir = feat_dir+key+df.Cohort[i]+p_df.ID[i]
            if os.path.exists(output_dir+'.feat'):
                if verb:
                    print('FEAT already exists for', df.Cohort[i]+p_df.ID[i])
                if over:
                    if verb:
                        print('Overwriting')
                    subprocess.Popen(['rm', '-rf', output_dir+'.feat'])
                else:
                    skip = True
            if not skip:
                to_write = stringTemp[:]
                # print(to_write)
                to_write = to_write.replace("%%OUTPUT_DIR%%",'"'+output_dir+'"')
                to_write = to_write.replace("%%VOLUMES%%",'"'+str(df.Dimension[i])+'"')
                to_write = to_write.replace("%%TR%%",'"'+str(df.TR[i])+'"')
                to_write = to_write.replace("%%BOLD_FILE%%",'"'+df.BOLD_path[i]+'"')
                to_write = to_write.replace("%%FS_T1%%",'"'+df.T1_path[i]+'"')
                to_write = to_write.replace("%%O2_CONTRAST%%",'"'+df.ETO2[i]+'"')
                to_write = to_write.replace("%%CO2_CONTRAST%%",'"'+df.ETCO2[i]+'"')
    
                ds_path = feat_dir+'design_files/'+key+df.ID[i]+'.fsf'
                with open(ds_path, 'w+') as outFile:
                    outFile.write(to_write)
    
                msg = False
                spin = '|/-\\'
                cursor = 0
    #                while not any(v is None for v in pro):
    #                    if verb:
    #                        if not msg:
    #                            print("There are 10 FEATs currently running. Waiting for at least one to end.")
    #                            msg = True
    #                        else:
    #                            sys.stdout.write(spin[cursor])
    #                            sys.stdout.flush()
    #                            cursor += 1
    #                            if cursor >= len(spin):
    #                                cursor = 0
    #                    for process in pro:
    #                        if process.poll()!= None:
    #                            pro[pro.index(process)] = None
    #                            break
    #                    if verb:
    #                        if msg:
    #                            time.sleep(0.2)
    #                            sys.stdout.write('\b')
    #                            
    #                index = pro.index(None)
                # os.spawnlp(os.P_NOWAIT, 'feat', 'feat', ds_path, '&')
                if verb:
                    print('Starting FEAT')
    #                pro[index] = subprocess.Popen(['feat', ds_path, '&'])
                process = subprocess.Popen(['feat', ds_path, '&'])
                time.sleep(0.3)
                
                if verb:
                    while(process.poll() == None):
                        sys.stdout.write(spin[cursor])
                        sys.stdout.flush()
                        cursor += 1
                        if cursor >= len(spin):
                            cursor = 0
                        time.sleep(0.2)
                        sys.stdout.write('\b')
                    
            feat_output_dir = feat_dir+key+p_df.Cohort[i]+p_df.ID[i]+'.feat/'
            
            cz1 = pd.read_csv(feat_output_dir+'cluster_zstat1_std.txt', sep='\t', usecols=['-log10(P)', 'Z-MAX', 'COPE-MEAN'])
            cz1['ID'] = p_df.Cohort[i]+p_df.ID[i]
            cz1['type'] = key
            cz1 = cz1[['ID', 'type', '-log10(P)', 'Z-MAX', 'COPE-MEAN']]
            
            cz2 = pd.read_csv(feat_output_dir+'cluster_zstat2_std.txt', sep='\t', usecols=['-log10(P)', 'Z-MAX', 'COPE-MEAN'])
            cz2['ID'] = p_df.Cohort[i]+p_df.ID[i]
            cz2['type'] = key
            cz2 = cz2[['ID', 'type', '-log10(P)', 'Z-MAX', 'COPE-MEAN']]
            
            build = cz1.merge(cz2, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
            
            O2_mask_dir_path = feat_output_dir+'cluster_mask_zstat1.nii.gz'
            CO2_mask_dir_path = feat_output_dir+'cluster_mask_zstat2.nii.gz'
            
            if os.path.exists(feat_output_dir+'fq_O2'):
                if verb:
                    print('O2 featquery already exists for', p_df.Cohort[i]+p_df.ID[i])
                if over:
                    if verb:
                        print('Overwriting')
                    subprocess.run(['featquery', '1', feat_output_dir, '1', 'stats/cope1', 'fq_O2', '-p', '-s', O2_mask_dir_path])
            else:
                if verb:
                    print('Starting featquery for O2')
                subprocess.run(['featquery', '1', feat_output_dir, '1', 'stats/cope1', 'fq_O2', '-p', '-s', O2_mask_dir_path])
                
            O2 = feat_output_dir+'fq_O2/'
            fq1 = pd.read_csv(O2+'report.txt', sep='\t| ', header=None, usecols=[5], engine='python')
            fq1 = fq1.rename(columns={5 : 'mean'})
            fq1['ID'] = p_df.Cohort[i]+p_df.ID[i]
            fq1['type'] = key
            fq1 = fq1[['ID', 'type', 'mean']]
            build = build.merge(fq1, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
            
            
            if os.path.exists(feat_output_dir+'fq_CO2'):
                if verb:
                    print('CO2 featquery already exists for', p_df.Cohort[i]+p_df.ID[i])
                if over:
                    if verb:
                        print('Overwriting')
                    subprocess.run(['featquery', '1', feat_output_dir, '1', 'stats/cope2', 'fq_CO2', '-p', '-s', CO2_mask_dir_path])
            else:
                if verb:
                    print('Starting featquery for CO2')
                subprocess.run(['featquery', '1', feat_output_dir, '1', 'stats/cope2', 'fq_CO2', '-p', '-s', CO2_mask_dir_path])
                
            CO2 = feat_output_dir+'fq_CO2/'
            fq2 = pd.read_csv(O2+'report.txt', sep='\t| ', header=None, usecols=[5], engine='python')
            fq2 = fq2.rename(columns={5 : 'mean'})
            fq2['ID'] = p_df.Cohort[i]+p_df.ID[i]
            fq2['type'] = key
            fq2 = fq2[['ID', 'type', 'mean']]
            build = build.merge(fq2, on=['ID', 'type'], suffixes=('_O2', '_CO2'))
            
            stats_df = pd.concat([stats_df, build])

#        while not all(v is None for v in pro):
#            if verb:
#                if not msg:
#                    print("Waiting for the remaining FEAT to finish")
#                    msg = True
#                else:
#                    sys.stdout.write(spin[cursor])
#                    sys.stdout.flush()
#                    cursor += 1
#                    if cursor >= len(spin):
#                        cursor = 0
#            for process in pro:
#                if process != None and process.poll()!= None:
#                    pro[pro.index(process)] = None
#                    break
#            if verb:
#                if msg:
#                    time.sleep(0.2)
#                    sys.stdout.write('\b')

    # run featquery
#        for i in range(len(p_df)):
#            feat_output_dir = feat_dir+key+p_df.Cohort[i]+p_df.ID[i]+'.feat'
#            if os.path.exists(feat_output_dir+'/featquery'):
#                if verb:
#                    print('featquery exists for', p_df.Cohort[i]+p_df.ID[i])
#                if over:
#                    if verb:
#                        print('Overwritting')
#                    subprocess.Popen(['rm', '-rf', feat_output_dir+'/featquery'])
#                else:
#                    continue
#            O2_mask_dir_path = feat_output_dir+'/cluster_mask_zstat1.nii.gz'
#            CO2_mask_dir_path = feat_output_dir+'/cluster_mask_zstat2.nii.gz'
#            if verb:
#                print('Running featquery on', p_df.Cohort[i]+p_df.ID[i])
#            subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope1', 'fq_O2', '-p', '-s', O2_mask_dir_path])
#            subprocess.Popen(['featquery', '1', feat_output_dir, '1', 'stats/cope2', 'fq_CO2', '-p', '-s', CO2_mask_dir_path])
#           os.spawnlp(os.P_NOWAIT, 'featquery', 'featquery', '1', feat_output_dir, '1', 'stats/cope1', 'ftqry_O2', '-p', '-s', '-w', O2_mask_dir_path, '&')
#           os.spawnlp(os.P_NOWAIT, 'featquery', 'featquery', '1', feat_output_dir, '1', 'stats/cope2', 'ftqry_CO2', '-p', '-s', '-w', CO2_mask_dir_path, '&')

stats_df.reset_index(drop=True)

if any([r2_dic[key] for key in r2_dic]):
    r2 = pd.DataFrame(r2_dic)
    print(r2.head())
    print(stats_df.head())
    stats_df = stats_df.merge(r2, on=['ID', 'type'], suffixes=('_O2', '_CO2'))


stats_df.to_csv(path+'stats_data.csv', sep='\t', index=False)

if verb:
    print('============== Script Finished ==============')





#
