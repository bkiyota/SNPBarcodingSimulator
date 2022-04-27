#!/bin/bash
#$ -S /bin/sh
#$ -cwd
#$ -j y

# qsub -l s_vmem=128G,mem_req=128G snp_profile_simulation.sh 0.0045 10 20 2 1

source ~/setenv.sh

cmd=`which python`
 
transcriptsPath=/home/brett/work/TrajectoryInference/SNPBarcodingSimulator/input/transcripts.csv
countsPath=/home/brett/work/TrajectoryInference/SNPBarcodingSimulator/input/LV_counts_10hpf.csv
heterozygosity=$1
n_embryos=$2
mean_cells=$3
sd_cells=$4
trial=$5
outDir=$6

${cmd} snp_profile_simulation.py \
-t $transcriptsPath \
-c $countsPath \
-v $heterozygosity \
-n $n_embryos \
-m $mean_cells \
-s $sd_cells \
-i $trial \
-o $outDir
