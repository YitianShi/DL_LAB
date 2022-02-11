#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=Yitian_Shi
#SBATCH --output=job_name-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

# Activate everything you need
module load cuda/10.1
# Run your python code
python3 main.py