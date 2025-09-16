#!/bin/bash

# Set the directory where your meshes or input files are
DIRECTORY="/your/path/to/processed_meshes"

# Count the number of files (e.g., one job per mesh file)
NUMBEROFRUNS=$(ls -1 "$DIRECTORY"/*.vol.gz | wc -l)

# Optional: print the number of jobs for logging
echo "Submitting $NUMBEROFRUNS jobs..."

start=$(date +%s)

# Submit array job to SLURM (adjust range to match file count)
sbatch --array=0-$(($NUMBEROFRUNS - 1)) runsingleshell.SLURM

end=$(date +%s)
echo "Submission completed in $(($end - $start)) seconds."

