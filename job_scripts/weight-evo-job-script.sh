#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=3:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=501
#SBATCH --mem=230G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=f.h.schijlen@student.tudelft.nl

module use /opt/insy/modulefiles

source ./env/bin/activate
python3 ./main.py
