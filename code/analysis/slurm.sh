#!/usr/bin/env bash

# Run from BIDS code/preprocessing directory: sbatch slurm_mriqc.sh

# Name of job?
#SBATCH --job-name=iluvslrmLOL

# Where to output log files?
# make sure this logs directory exists!! otherwise the script won't run
#SBATCH --output='/jukebox/graziano/coolCatIsaac/mei/code/analysis/slurm_out/isc-%A_%a.log'

# Set partition
#SBATCH --partition=all

# How long is job?
#SBATCH -t 5:00:00

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=4 

# Update with your email 
#SBATCH --mail-user=isaacrc@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# member to reinsert sbatch if needed boi
# --array=0-1
#printf -v roi $SLURM_ARRAY_TASK_ID
#echo "$SLURM_ARRAY_TASK_ID"

# Remove modules because Singularity shouldn't need them
echo "Purging modules"
module purge

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
date

#Run script located in directory ##
export PYTHONUNBUFFERED=1

# PARTICIPANT LEVEL
echo "leggo"
module load pyger/0.11.0
cd /jukebox/graziano/coolCatIsaac/mei/code/analysis

## leggo ##
python isc-salience.py

echo "Finished"
date
