#!/bin/bash
#SBATCH -J zeng_da
#SBATCH -p kshdexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --gres=dcu:4
#SBATCH --mem 90G
##SBATCH --ntasks-per-node=1

date
export PYTHONPATH=/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation:/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/dtb/python
module rm compiler/rocm/2.9
module load compiler/rocm/4.0.1
mpirun python DA_Voxel.py \
  --ip=11.5.4.2:50051 \
  --block_path='/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/model_dynamics/block_path/inner_25/single' \
  --write_path='../da_simulation/da_region_brain_with_new_method'\
  --T=200 \
  --bold_sigma=1e-8 \
  --hp_sigma=0.25 \
  --ensembles=100 \
  --noise_rate=0.0003
