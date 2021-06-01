#!/bin/sh
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=2
#SBATCH --tasks-per-node=32
#SBATCH --constraint=haswell

python3 run_map.py
