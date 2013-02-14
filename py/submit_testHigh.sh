#!/bin/sh
#Run qsub -l h_rt=36:00:00
#$ -pe orte 40
#$ -cwd
#$ -V
#$ -R y
#$ -r y
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/gsl-1.15/lib/:/usr/lib64
export MPI=/usr/local/openmpi/gcc/x86_64
export PATH=${MPI}/bin:${PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MPI}/lib
export OMP_NUM_THREADS=1
mpirun -x PYTHONPATH /home/bovy/local/bin/python pixelFitDF.py ../realDF/realDFFitK_dfeh0.1_dafe0.05_dpdiskplhalofixbulgeflatwgasalt_altOptHighAlt5000_fitdvt_staeckel_singles_22.sav --sample=k --staeckelRmax=5.000000 --staeckelnE=50 --fitdvt --aAmethod=staeckel --snmin=15.000000 --staeckeldelta=0.450000 --nsamples=5000 --minndata=100 --select=program --mcsample --singlefeh=-0.35000 --ext=png --seed=1 --aAzmax=1.000000 --aAnLz=31 --aARmax=5.000000 --nmc=1000 --nscale=1.000000 --nv=201 --indiv_brightlims --potential=dpdiskplhalofixbulgeflatwgasalt --height=1.100000 --staeckelnLz=60 --dfeh=0.100000 --ninit=0 --dafe=0.050000 --ngl=20 --aAnEr=31 --singleafe=0.125 --aAnR=16 --aAnEz=16 --index=0 --multi=1 --nmcv=1000 --dfmodel=qdf --staeckelnpsi=50 --nmcerr=30 --starthigh --mpi
