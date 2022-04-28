#!/bin/bash
#SBATCH -J main
#SBATCH -p kshdexclu04
#SBATCH -N 100
#SBATCH -n 100
#SBATCH --cpus-per-task 8
#SBATCH --gres=dcu:4
#SBATCH --mem 90G

date
export PYTHONPATH=/public/home/ssct004t/project/zenglb/CriticalNN
module switch compiler/rocm/4.0.1
mpirun -np 100 python run_grid_search.py \
  --block_path='/public/home/ssct004t/project/zenglb/CriticalNN/data/small_block' \
  --write_path='/public/home/ssct004t/project/zenglb/CriticalNN/data/grid_search'
date




