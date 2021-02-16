#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=101
#SBATCH --mem=55G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=f.h.schijlen@student.tudelft.nl

module use /opt/insy/modulefiles

source ./env/bin/activate
python3 ./main.py
