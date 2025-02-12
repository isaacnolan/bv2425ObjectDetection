#!/bin/bash
#SBATCH -o "NVIDIA.txt"
#SBATCH --job-name=HyperParameterTuning
#SBATCH --nodes=1 --ntasks-per-node=28 --gpus-per-node=1 --gpu_cmode=shared
#SBATCH --time=8:00:00
#SBATCH --account=PAS2926
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

module load miniconda3/23.3.1-py310 cuda/12.3.0

source activate ultralytics-env

echo "Python executable: $(which python)"

nvidia-smi
