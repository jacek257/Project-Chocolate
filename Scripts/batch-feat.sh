#!/bin/bash

###############################################################
# PROCESS: Runs FEAT analysis, FEATquery, and gets mean whole
#          brain CVR and CBV
# HOW TO USE:
# ./batch-feat.sh [/path/to/subject_dir]
#
# Ex. ./batch-feat.sh /home/labrat/SH001
###############################################################

feat_analysis () {
    local dir=$1
    local bold=$2
    # Get SubjectID_Date and paths to BOLD, T1, ETCO2, ETO2 files
    sid=${dir##*/}
    date=${bold##*_}
    sid_date=$sid"_"${date:0:8}
    # Use brainmask as T1 or BET original T1 for registration
    if [ -f $SUBJECTS_DIR/$sid_date/brainmask.mgz ]; then
        mri_convert -i $SUBJECTS_DIR/$sid_date/brainmask.mgz -o $dir/fs.t1.nii.gz
        t1=$dir/fs.t1.nii.gz
    else
        if [[ -f $dir/t1_brain.nii.gz ]]; then
            echo "[0] Grabbing BET'd T1" >> $dir/pipeline_log.txt
        else
            t1=`(find $dir -name "*SAG_FSPGR_3D*.nii.gz" -o -name "*SAG_FSPGR_3D*.nii") | head -n 1`
            echo "[0] Running FSL BET: $t1" >> $dir/pipeline_log.txt
            bet $t1 $dir/t1_brain.nii.gz -f 0.6 -R
        fi
        t1=$dir/t1_brain.nii.gz
    fi
    etdir="/home/labrat/Dropbox/EndTidal_Recordings"
    co2=$(find $etdir -name "*$sid*ETCO2*.txt") 
    o2_=$(find $etdir -name "*$sid*ETO2*.txt") 
    out="$dir/${date}_bold"
    while [[ -d $out ]]; do
        out=$out"+"
    done

    # Output variables to pipeline_log.txt
    echo "[SID]" $sid_date >> $dir/pipeline_log.txt
    echo "[DIR]" $dir >> $dir/pipeline_log.txt
    echo "[OUTDIR] $out" >> $dir/pipeline_log.txt
    echo "[1] Files for FSL FEAT analysis" >> $dir/pipeline_log.txt
    echo "[BOLD] $bold" >> $dir/pipeline_log.txt
    echo "[T1] $t1" >> $dir/pipeline_log.txt
    echo "[ETCO2] $co2" >> $dir/pipeline_log.txt
    echo "[ETO2] $o2" >> $dir/pipeline_log.txt
    
    # Match strings for design.fsf file
    bold_str="BOLD_FILE"
    t1_str="T1_FILE"
    co2_str="ETCO2_FILE"
    o2_str="ETO2_FILE"
    out_str="OUT_DIR"

    # Run FEAT analysis if all files are there
    if [[ $bold != "" ]] || [[ $co2 != "" ]] || [[ $o2 != "" ]]; then
        echo "[2] Run FEAT analysis"
        sed "s!$bold_str!$bold!g; s!$t1_str!$t1!g; s!$co2_str!$co2!g; \
            s!$o2_str!$o2!g; s!$out_str!$out!g" \
            /home/labrat/Scripts/design_template.fsf > $dir/design.fsf
        feat $dir/design.fsf
    else
        echo "[2] Cannot run FEAT analysis"
    fi
}


for dir in $@; do
    echo "[FSL FEAT] Running batch-feat.sh" >> $dir/pipeline_log.txt
    for bold in $(find $dir -name "*BOLD*.nii.gz" -o -name "*BOLD*.nii"); do
        feat_analysis "$dir" "$bold"
    done
    echo >> $dir/pipeline_log.txt
done
wait
