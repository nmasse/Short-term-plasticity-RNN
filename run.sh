#!/bin/bash
#SBATCH --job-name=rnn_stp
#SBATCH --output=rnn_stp.out
#SBATCH --error=rnn_stp.err
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --time=035:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=2000

module load python/3.5.2
python run_all_models.py
