#!/bin/bash
#SBATCH --job-name=bart-mg
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.40gb:2

#SBATCH --output=gpu_job-t5-base_.out
#SBATCH --error=gpu_job-t5-base_.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=80G
#SBATCH --time=1-23:59:00

#Load needed modules

source ../T5/august/bin/activate

module load gnu10
module load python
pip install transformers
pip install sentencepiece
pip install torch
pip install numpy
module load transformers
module load torch
module load numpy

deepspeed --master_port 2433 main.py configs/train_deepspeed.json


