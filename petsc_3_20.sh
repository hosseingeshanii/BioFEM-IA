#!/bin/bash
ml GCC/12.3.0  OpenMPI/4.1.5
ml PETSc/3.20.3
export PATH="/scratch/user/hgeshani/.conda/envs/pyt38/bin:$PATH"
export PYTHONPATH="/scratch/user/hgeshani/.conda/envs/pyt38/lib/python3.8/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/scratch/user/hgeshani/.conda/envs/pyt38/lib:$LD_LIBRARY_PATH"
