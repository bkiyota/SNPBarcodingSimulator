#!/bin/bash

# for heterozygosity in `seq 0.000015 0.000001 0.00006`
# do
#     for trial in `seq 1 1 3`
#     do
#         qsub -l s_vmem=32G,mem_req=32G \
#         snp_profile_simulation.sh \
#         $heterozygosity \
#         10 \
#         20 \
#         2 \
#         $trial
#     done
# done

# for n_embryos in `seq 2 2 100`
# do
#     for trial in `seq 1 1 3`
#     do
#         qsub -l s_vmem=32G,mem_req=32G \
#         snp_profile_simulation.sh \
#         0.0015 \
#         $n_embryos \
#         15 \
#         5 \
#         $trial
#     done
# done

# for n_embryos in `seq 2 2 100`
# do
#     for trial in `seq 1 1 3`
#     do
#         qsub -l s_vmem=32G,mem_req=32G -N bkSNP_Lv${n_embryos} \
#         snp_profile_simulation.sh \
#         0.000528 \
#         $n_embryos \
#         15 \
#         5 \
#         $trial \
#         /home/brett/work/TrajectoryInference/SNPBarcodingSimulator/simulation/analysis/num_embryo_analysis_repeat
#     done
# done

for n_embryos in `seq 200 100 1000`
do
    qsub -l s_vmem=32G,mem_req=32G -N bkSNP_Lv${n_embryos} \
    snp_profile_simulation.sh \
    0.000528 \
    $n_embryos \
    15 \
    5 \
    1 \
    /home/brett/work/TrajectoryInference/SNPBarcodingSimulator/simulation/analysis/num_embryo_analysis_large
done