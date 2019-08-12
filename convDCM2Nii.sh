#!/bin/bash
# args = $@

# traverse() {
#     for dir in $(find $PWD$inputdir -maxdepth 1 -mindepth 1 -type d)
#     do
#     	echo "DIR: $dir"
# 	for serdir in $(find $dir -maxdepth 2 -mindepth 2 -type d)
# 	do
# 		echo "SERDIR: $serdir"
# 		echo "inside serdir"
# 	    filename=""
# 	    for dcmfile in $serdir/*.dcm
# 	    do
# 		filename=`python read_dicom.py $dcmfile`
# 		filename=${filename// /_}
# 		echo "filename :$filename"
# 		if [ $filename != "" ]
#           	then
#               	    # (cd $dir && mkdir $filename &&
#               	    # 	/Applications/MRIcron/dcm2nii -i n -p y -v y -o $dir/$filename $serdir)
#               	    (cd $dir && mkdir $filename &&
#               	    	/Users/baymac/Desktop/Processing_code/MRIcroGL/dcm2niix -f %n_%p_%t -o $dir/$filename $serdir)
#            	fi
# 		break
# 	    done
# 	done
#     done
#     return
# }


traverse() {
    for dir in $(find $PWD$inputdir -maxdepth 1 -mindepth 1 -type d)
    do
    	DEST="/Users/baymac/Desktop/Data_todo/NII"
    	echo "DIR: $dir"
    	ID=$(basename $dir)
    	echo "ID: $ID"
    	mkdir -p $DEST/$ID
		for serdir in $(find $dir -mindepth 2 -type d)
		do
			echo "SERDIR: $serdir"
			D2=$(basename $serdir)
			PARENT=$(dirname $serdir)/
			echo "Parent:  $PARENT"
			# DEST="$DEST/$ID/$D2"
			# echo "DEST: $DEST/$ID/$D2"
			mkdir $DEST/$ID/$D2
			# echo "$PWD"
			# /Users/baymac/Desktop/Processing_code/MRIcroGL/dcm2niix -f %i_%p_%t -o $DEST/$ID/$D2/ $serdir
			dcm2niix -f %i_%p_%t -o $DEST/$ID/$D2/ $serdir
	    done
    done
    return
}

inputdir=$1
echo "Running image conversion script."
if [ "`whoami`" = "baymac" ]
then
    traverse
fi
echo "-----------!!!!!Finished image conversion script!!!!--------------"
