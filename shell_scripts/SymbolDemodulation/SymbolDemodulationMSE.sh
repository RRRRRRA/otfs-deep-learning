#!/bin/bash
#SBATCH --job-name=SymbolDemodulation
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jturley1@umbc.edu
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --error=SymbolDemodulation.err
#SBATCH --output=SymbolDemodulation.out

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"

conda activate 675

python ~/gokhale_user/675/project/python_scripts/SymbolDemodulationMSE.py
