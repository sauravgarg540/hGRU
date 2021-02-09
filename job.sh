#!/bin/bash
# SBATCH --ntasks=1
# SBATCH --time=00:01:00
# SBATCH --gres=gpu:4
# SBATCH --mail-type=ALL
# SBATCH --mail-user=vorname.nachname@uni-ulm.de
echo "Starting job"
python main.py 