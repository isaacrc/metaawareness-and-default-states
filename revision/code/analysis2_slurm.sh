#!/bin/bash
#SBATCH --job-name=mei_wsfc_trans
#SBATCH --output=/jukebox/graziano/coolCatIsaac/mei/revision/results/connectivity/slurm_%A_%a.out
#SBATCH --error=/jukebox/graziano/coolCatIsaac/mei/revision/results/connectivity/slurm_%A_%a.err
#SBATCH --array=1-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=all

echo "Starting SLURM job $SLURM_JOB_ID, array task $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Date: $(date)"

# Activate environment
source /jukebox/pkgs/PYGER/base/etc/profile.d/conda.sh
conda activate mei_

SCRIPT=/jukebox/graziano/coolCatIsaac/mei/revision/code/analysis2_connectivity_transition.py
LOG_DIR=/jukebox/graziano/coolCatIsaac/mei/revision/results/connectivity

echo "Running transition $SLURM_ARRAY_TASK_ID with 10000 permutations"
python $SCRIPT --transition $SLURM_ARRAY_TASK_ID --n_perms 10000

echo "Done: $(date)"

# After all 3 finish, generate the combined figure
# (only runs when all 3 transition files exist)
if [ -f "$LOG_DIR/transition_1to2_wsfc.npy" ] && \
   [ -f "$LOG_DIR/transition_2to3_wsfc.npy" ] && \
   [ -f "$LOG_DIR/transition_3to4_wsfc.npy" ]; then
    echo "All transitions complete — generating combined figure"
    python $SCRIPT --transition all --n_perms 0  # just plot, no recompute
fi
