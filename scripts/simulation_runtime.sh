#!/bin/bash

dirPath=/home/brett/work/TrajectoryInference/SNPBarcodingSimulator/simulation/analysis/qsub_outfiles
runtimePath=${dirPath}/simulation_runtime.csv
rm -f $runtimePath
echo "label,id,start_time,end_time" >> $runtimePath
for filePath in ${dirPath}/bkSNP_Lv*
do
    label=$( basename $filePath | cut -d. -f1 )
    jobID=$( basename $filePath | cut -d. -f2 | cut -do -f2 )
    s0=$( qreport -j $jobID | grep "start_time" )
    e0=$( qreport -j $jobID | grep "end_time" )
    start=$( echo $s0 | tr -s " " | rev | cut -d " " -f 1 | rev )
    end=$( echo $e0 | tr -s " " | cut -d " " -f 2 )

    # start0=$( qreport -j $jobID | grep "start_time" | cut -d- -f2 | awk '{$1=$1;print}' )
    # end0=$( qreport -j $jobID | grep "end_time" | cut -d- -f2 | awk '{$1=$1;print}' )

    # start1=$( echo $test0 | cut -d' ' -f1 )
    # end1=$( echo $test1 | cut -d' ' -f1 )
    
    echo "${label},${jobID},${start},${end}" >> $runtimePath
done