#!/bin/bash

traverse() {
    for dir in $(find $PWD$inputdir -maxdepth 1 -mindepth 1 -type d)
    do
    	DEST="~/Desktop/Data_todo/NII"
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

mkdir -p ~/Desktop/Data_todo
mkdir -p ~/Desktop/Data_todo/NII

traverse

echo "-----------!!!!!Finished image conversion script!!!!--------------"
