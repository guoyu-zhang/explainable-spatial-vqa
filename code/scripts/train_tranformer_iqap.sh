\#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N train_model
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=80G
#$ -o ../logs/train_transformer_iqap/train.log
#$ -e ../logs/train_transformer_iqap/train.err
#$ -q gpu
#$ -pe gpu-a100 1
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

> $(pwd)/../logs/train_transformer_iqap/train.err
> $(pwd)/../logs/train_transformer_iqap/train.log

# Initialise the environment modules
. /etc/profile.d/modules.sh
export XDG_CACHE_HOME="/exports/eddie/scratch/s1808795/.cache"

# Load Python
module load cuda
module load anaconda
conda activate diss

# Run the program
python /home/s1808795/vqa/code/train_transformer_iqap.py 

# deactivate



