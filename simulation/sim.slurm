#!/bin/bash
#SBATCH -J nn_49
#SBATCH -p kshdexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --gres=dcu:4
#SBATCH --mem 90G
#SBATCH --exclude=e04r1n03
##SBATCH --ntasks-per-node=1

 
date
export PYTHONPATH=/public/home/ssct004t/project/zenglb/CriticalNN:/public/home/ssct004t/project/zenglb/CriticalNN/dtb/python
module switch compiler/rocm/4.0.1
mpirun python cuda_simulate.py \
  --ip=11.5.4.11:50051 \
  --block_path="/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/model_dynamics/block_path/inner_25/single" \
  --write_path='/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/da_simulation/da_region_brain_with_new_method/' \
date
