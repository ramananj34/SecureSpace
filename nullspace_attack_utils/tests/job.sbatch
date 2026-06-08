#!/bin/bash
#SBATCH --job-name=nullspace_longGE
#SBATCH -p gpu
#SBATCH -n 4
#SBATCH --mem 64G
#SBATCH --gres=gpu:h100:1
#SBATCH -t 0:30:00
#SBATCH --mail-user=jramanan@colgate.edu
#SBATCH --mail-type=ALL

source /home/jramanan/miniconda3/bin/activate
conda activate securespace

cd /home/jramanan/SecureSpace

python nullspace_attack/tests/verify_null_space_long_ge_gpu.py 2>&1 | tee nullspace_longGE_log.txt