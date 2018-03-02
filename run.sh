#!/bin/bash
#SBATCH --job-name=rnn_stp
#SBATCH --output=rnn_stp.out
#SBATCH --error=rnn_stp.err
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16000

module load python/3.5.2
python run_all_models.py $1
