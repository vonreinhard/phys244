# Credit

## OpenMP code author

Mark Harris - NVIDIA Corporation

Andreas Goetz - SDSC: OMP code and minor modifications

## CUDA rrewrite author

TzuKao Wang - UCSD Grad Student

# Description


# Files

Miscellaneous:  
`timer.h` - simple timing code

Original:  
`md_openmp.c`   - The serial C code use OpenMP 

Our Work:  
`md_openmp.cu`   - CUDA parallel C code  

Script:  
`CUDA.sb` - A simple script to running cuda version of our work.

`md-omp.sb` - A simple script to running openmp version of our work.
# OpenMP
Compile:  
Using the PGI compiler

    # OpenMP serail command
    module purge
    module load slurm
    module load gpu
    module load pgi
    export OMP_NUM_THREADS=16
    pgcc -fast -mp -Minfo=mp -o md_openmp.x md_openmp.c
    ./md_openmp.x

# CUDA part
To run CUDA version interactively, you need to do following commands: 

    # CUDA serial command
    module purge
    module load slurm
    module load gpu
    module load cuda
    nvcc CUDA/md_openmp.cu -o md.o
    ./md.o

