#!/bin/bash

#SBATCH --account=elaine

# jobname to be used for the job allocation
#SBATCH -J Astec 

#User notify on events (BEGIN,END,FAIL,REQUEUE,ALL)
#SBATCH --mail-type=ALL

# User to notify
#SBATCH --mail-user=


# redirects error to file.err (%A is replaced by the job ID and %a with the array index.)
#SBATCH -e job-%A_%a.err

# redirect output to file.out.
#SBATCH -o job-%A_%a.out


# specify partition, i.e. compute
# In general most of the jobs should be scheduled in the partition "compute". 
# For testing purpose a partition "test" which is also the default should be used.
# "test" has a short runtime but a higher possibilety to start.
#
#SBATCH -p compute

# export environment variables
# SBATCH --export=ALL, OMP_NUM_THREADS

#Number of Nodes
# Have in mind, partition "compute" needs at least 2 nodes!
#SBATCH --nodes=2 
#SBATCH --ntasks=80


#request <x>MB real memory per node, max mem is 64000 per node
# compare: #sinfo -o "%n %m %c" | head
#alternative to --mem , --mem-per-cpu=<mem> could be used

#request 12 hours run time
# for now wait how long it will take
#SBATCH --time=48:0:0

#configure your shell environment using modules

#working directory
#SBATCH --chdir=/data/bo8482/cgal_mesh/new_elegans_run/very_low_cond_near


# load module which configures everything 
module load intel/19.1.0 gcc/9.3.0 openmpi/gcc.9/3.1.6 python/3.8.2 petsc-3.20.1-complex 
source ~/ngs24/bin/activate

# Lets put some info in job.out
echo "Job was started on Nodes: $SLURM_NODELIST"
echo "We are on Node: $SLURM_NODEID"
echo "JobID was: $SLURM_JOB_ID"
echo "User id used is: $USER"
echo "Submit Directory is $SLURM_SUBMIT_DIR"

echo "Number of tasks $SLURM_NTASKS"
export OMP_NUM_THREADS=1

echo "Job ID $SLURM_ARRAY_TASK_ID"

mpirun -np 80 python3 -u solver.py ${SLURM_ARRAY_TASK_ID} &> SingleShell_${SLURM_ARRAY_TASK_ID}.log
