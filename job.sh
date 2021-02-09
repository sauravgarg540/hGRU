#!/bin/bash
# SBATCH --ntasks=1
# SBATCH --time=25:00:00
# SBATCH --gres=gpu:1
# SBATCH --mail-type=ALL
# SBATCH --mail-user=sauravgupta3108@gmail.com
python -W ignore dp_main.py 