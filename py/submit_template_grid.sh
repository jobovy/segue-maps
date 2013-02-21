#!/bin/sh
#Run qsub -l h_rt=36:00:00
#$ -pe orte 1
#$ -cwd
#$ -V
#$ -R y
#$ -r y
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/gsl-1.15/lib/:/usr/lib64
export MPI=/usr/local/openmpi/gcc/x86_64
export PATH=${MPI}/bin:${PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MPI}/lib
export OMP_NUM_THREADS=1
