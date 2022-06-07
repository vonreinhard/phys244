#!/bin/bash
#SBATCH --job-name="./md"
#SBATCH --output="cuda.%j.%N.out"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=96G
#SBATCH --account=csd453
#SBATCH --export=ALL
#SBATCH --gpus =1
#SBATCH -t 00:03:00




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

