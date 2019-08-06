'''
By: Austin Sophonsri

'''

import os,sys
import subprocess
import argparse
import pandas as pd

# instantiate the argument parser
parser = argparse.ArgumentParser()

# add positional argument
parser.add_argument("path", help="path of the folder that contains all the endtidal edit files")

# add optional arguments
parser.add_argument("-v", "--verbose", action='store_true', help="incrase output verbosity")
parser.add_argument("-g", "--graph", action='store_true', help='display graphs as it is generated')

#get the positional arguments
args = parser.parse_args()
path = args.path

verb = (True if args.verbose else False)
graph = (True if args.graph else  False)

###########set directories (TODO, automate)
nifti_dir = '/media/labrat/395Mount/FSL_work/SH_NII/'
freesurfer_t1_dir = '/media/labrat/395Mount/FSL_work/FS_SH/'


# make sure the path ends with '/'
if path[-1] != '/':
    path += '/'

# all grab all the .txt files in the endtidal folder
txt_files = [file for file in os.listdir(path) if file.endswith('edits.txt')]

print(txt_files[:5])

#separate patient ID and scan date and pack into tuple
p_df = pd.DataFrame({
                'Cohort':[f[0:2] for f in txt_files],
                'ID':[f[3:6] for f in txt_files],
                'Month':[f[11:13] for f in txt_files],
                'Day':[f[13:15] for f in txt_files],
                'Year':[f[7:11] for f in txt_files],
                'EndTidal_Path': [path+f for f in txt_files]
             })

print(p_df.head())
#create patient bold scan listdir (is a list not DataFrame)
patient_BOLDS_header = [p_df.Cohort[i]+p_df.ID[i]+'_BOLD_'+p_df.Year[i]+p_df.Month[i]+p_df.Day[i]
                    for i in range(len(p_df))]

print(patient_BOLDS_header[:5])

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

#drop all false conditional rows and conditional column
p_df = p_df[p_df.boldFS_exists != False].drop('boldFS_exists', axis = 1)

print('\n',p_df.head())

#get number of volumes
#run fslinfo and pipe -> python buffer -> replace \n with ' ' -> split into list -> choose correct index -> convert string to int
p_df["Dimension"] = [int(subprocess.run(['fslinfo' ,b_path], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n',' ').split()[9]) for b_path in p_df.BOLD_path]
#choose non-trivial bold series
p_df = p_df[p_df.Dimension >=150]
print('\n',p_df.head())

####run EndTidal Cleaning and return paths
#make endtidal folders

ET_dict = {'ETO2' : [], 'ETCO2' : [], 'ET_exists' : []}

for f_path in p_df.EndTidal_Path:
    if not os.path.exists(f_path[:-4]):
        os.mkdir(f_path[:-4])

    #perfrom cleaning and load into p_df
    #load data into DataFrame
    df = pd.read_csv(f_path, sep='\t|,', names=['Time', 'O2', 'CO2', 'thrw', 'away'], usecols=['Time', 'O2', 'CO2'], index_col=False, engine='python')
    #drop rows with missing cols
    df = df.dropna()

    #skip if DataFrame is empty
    if df.empty:
        os.rmdir(f_path[:-4])
        ET_dict['ETO2'].append('')
        ET_dict['ETCO2'].append('')
        ET_dict['ET_exists'].append(False)
        continue

    # need to scale CO2 data is necessary
    if df.CO2.max() < 1:
        df.CO2 = df.CO2 * 100

    #get fourier cleaned data
    #generate and store cleaned data paths
    print(df.O2[:5])
    print(df.CO2[:5])
    #save data

#run Feat










#
