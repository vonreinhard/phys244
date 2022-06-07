#!/bin/bash
#SBATCH --job-name="cuda"
#SBATCH --output="cuda_8192.%j.%N.out"
#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --account=csd453
#SBATCH --export=ALL
#SBATCH -t 00:30:00




module purge
module load slurm
module load gpu
module load cuda
## Example of OpenMP code running on a shared node
#SET the number of openmp thread
#
for i in 1
do
  echo "Test"
  ./md.o
  if [ $? -ne 0 ]; then
    echo "Run error."
    exit
  fi
done
#
echo "Normal end of execution."

