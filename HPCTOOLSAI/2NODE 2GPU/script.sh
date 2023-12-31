#!/bin/sh
#SBATCH -J jsimple       # Job name
#SBATCH -o jsimple.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e jsimple_job.o%j   # Name of stderr output file(%j expands to jobId)
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=4
#SBATCH -c 32
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -p short --qos=short


echo SLURM_NTASKS: $SLURM_NTASKS
echo SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK
echo SLURM_NNODES: $SLURM_NNODES
echo SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE

PS=1
WORKERS=$((SLURM_NTASKS-PS))



srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK  --resv-ports=$SLURM_NTASKS_PER_NODE -l python parallel.py $1 -ps $PS -workers $WORKERS
