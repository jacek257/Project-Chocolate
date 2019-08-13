[1mdiff --git a/nomeclature.py b/nomeclature.py[m
[1mindex 9b2fbbf..37a8fca 100644[m
[1m--- a/nomeclature.py[m
[1m+++ b/nomeclature.py[m
[36m@@ -24,20 +24,27 @@[m [mpath = args.path[m
 verb = (True if args.verbose else False)[m
 graph = (True if args.graph else  False)[m
 [m
[32m+[m[32m###########set directories (TODO, automate)[m
[32m+[m[32mnifti_dir = '/media/labrat/395Mount/FSL_work/SH_NII/'[m
[32m+[m[32mfreesurfer_t1_dir = '/media/labrat/395Mount/FSL_work/FS_SH/'[m
[32m+[m
[32m+[m
 # make sure the path ends with '/'[m
 if path[-1] != '/':[m
     path += '/'[m
 [m
 # all grab all the .txt files in the endtidal folder[m
[31m-txt_files = [file for file in os.listdir(path) if file.endswith('.txt')][m
[32m+[m[32mtxt_files = [file for file in os.listdir(path) if file.endswith('edits.txt')][m
[32m+[m
[32m+[m[32mprint(txt_files[:5])[m
 [m
 #separate patient ID and scan date and pack into tuple[m
 p_df = pd.DataFrame({[m
[31m-                'Cohort':[''.join(f[0:2]) for f in txt_files],[m
[31m-                'ID':[''.join(f[3:6]) for f in txt_files],[m
[31m-                'Month':[''.join(f[7:9]) for f in txt_files],[m
[31m-                'Day':[''.join(f[9:11]) for f in txt_files],[m
[31m-                'Year':[''.join(f[11:15]) for f in txt_files][m
[32m+[m[32m                'Cohort':[f[0:2] for f in txt_files],[m
[32m+[m[32m                'ID':[f[3:6] for f in txt_files],[m
[32m+[m[32m                'Month':[f[11:13] for f in txt_files],[m
[32m+[m[32m                'Day':[f[13:15] for f in txt_files],[m
[32m+[m[32m                'Year':[f[7:11] for f in txt_files][m
              })[m
 [m
 print(p_df.head())[m
[36m@@ -47,14 +54,35 @@[m [mpatient_BOLDS_header = [p_df.Cohort[i]+p_df.ID[i]+'_BOLD_'+p_df.Year[i]+p_df.Mon[m
 [m
 print(patient_BOLDS_header[:5])[m
 [m
[31m-#kinda the same thing as below[m
[31m-# BOLD_files = [file for it, file in enumerate(os.listdir(BOLD_path)) if (file.endswith('.nii') and file.startswith(patient_BOLDS_header[it]))][m
[32m+[m[32m#get bold and FS paths[m
[32m+[m[32mnii_paths = {'BOLD_path' : [], 'T1_path' : [], 'boldFS_exists': []}[m
[32m+[m[32mfor i in range(len(p_df)):[m
[32m+[m[32m    #if bold file doesnt exist then continue[m
[32m+[m[32m    if(not os.path.exists(nifti_dir + p_df.Cohort[i] + p_df.ID[i] +'/BOLD')):[m
[32m+[m[32m        nii_paths['BOLD_path'].append('')[m
[32m+[m[32m        nii_paths['T1_path'].append('')[m
[32m+[m[32m        nii_paths['boldFS_exists'].append(False)[m
[32m+[m[32m        continue[m
[32m+[m
[32m+[m[32m    #get all matching files[m
[32m+[m[32m    b_files = [file for file in os.listdir(nifti_dir + p_df.Cohort[i] + p_df.ID[i] +'/BOLD/') if file.endswith('a.nii')][m
[32m+[m[32m    fs_files = [file for file in os.listdir(freesurfer_t1_dir) if file == p_df.Cohort[i]+p_df.ID[i]+'_FS_T1.nii.gz'][m
[32m+[m
[32m+[m[32m    #select file[m
[32m+[m[32m    nii_paths['BOLD_path'].append(b_files[0] if len(b_files) > 0 else '')[m
[32m+[m[32m    nii_paths['T1_path'].append(fs_files[0] if len(fs_files) > 0 else '')[m
[32m+[m[32m    nii_paths['boldFS_exists'].append(len(b_files) > 0 and len(fs_files)>0)[m
[32m+[m
[32m+[m
[32m+[m[32m#append bold, FS paths, and conditional to p_df[m
[32m+[m[32mp_df = pd.concat((p_df, pd.DataFrame(nii_paths)), axis=1)[m
 [m
[31m-#but construct paths from patient dataframe[m
[31m-#so a multi line for loop may be necessary[m
[32m+[m[32m#drop all false conditional rows and conditional column[m
[32m+[m[32mp_df = p_df[p_df.boldFS_exists != False].drop('boldFS_exists', axis = 1)[m
 [m
[31m-#Bold_files =[m
[32m+[m[32mprint(p_df.sort_values('ID'))[m
 [m
[32m+[m[32m#run EndTidal Cleaning and return paths[m
 [m
 [m
 [m
