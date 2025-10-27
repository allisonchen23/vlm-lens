#!/bin/bash
#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8    # Specify the number of CPUs your task will need.
#SBATCH --mem=50G             # memory
#SBATCH -o slurm_logs/%j_%x_out.txt         # send stdout to outfile
#SBATCH -e slurm_logs/%j_%x_err.txt         # send stderr to errfile
#SBATCH -t 72:00:00           # time requested in hour:minute:second
#SBATCH --mail-user=allisonchen@princeton.edu
#SBATCH --mail-type=ALL

# Initialize conda for this script
eval "$(conda shell.bash hook)"
conda activate vlm-lens-base
cd notebooks
python tll.py