#!/bin/bash -l

# Batch script to run a GPU job under SGE.

# Request a number of GPU cards (2 is the maximum)
#$ -l gpu=1

# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=23:00:0

# Request RAM size (must be an integer followed by M, G, or T)
#$ -l mem=24G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=10G

# Set the name of the job.
#$ -N H48d_GCL_RMI_Occ

module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3
module load pytorch/1.11.0/gpu
source ~/sw/python/seg/bin/activate
cd ~/dev/repo/GenCL/
bash scripts/thermalFaceDB/hrnet/run_h_48_d_8_hrnet_gcl_rmi_train_occ.sh train hrnet_gcl_rmi_occ ~/dev/data/ThermalFaceDBx340 ~/dev/data/ThermalFaceDBx340