#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --ntasks-per-socket=2
#SBATCH --constraint="type_d|type_c"
#SBATCH --output=mpi%j.log

echo "Date = $(date)"
echo "Host = $(hostname -s)"
echo "WorkDir = $(pwd)"
echo ""
echo "Number of Nodes Allocated = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated = $SLURM_NTASKS"
echo "Numbers of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module load openmpi

srun /home/admitrakova/mpi < big_matrix_1.txt
