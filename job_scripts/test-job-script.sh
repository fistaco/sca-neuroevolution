#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:00:05
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0.1G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=f.h.schijlen@student.tudelft.nl

module use /opt/insy/modulefiles

source ./env/bin/activate
python3 ./test.py
