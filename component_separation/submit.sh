#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=5

module load python
module load 
source activate myenv
python hello-world.py