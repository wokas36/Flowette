#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --account=OD-239291

module load miniconda3

cd /scratch3/wij023/projects/flowex/
source activate /scratch3/wij023/projects/virtual_env/graphex_fm
python main.py