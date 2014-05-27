#!/bin/bash
# The name of the job, can be anything, simply used when displaying the list of running jobs
#$ -N R_w3_s1.py
# Combining output/error messages into one file
#$ -j y
# Set memory request:
#$ -l vf=2G
# Set walltime request:
#$ -l h_rt=72:00:00
# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username
#$ -cwd
# then you tell it retain all environment variables (as the default is to scrub your environment)
#$ -V
# Now comes the command to be executed
module load python
python evalTaggerR_w3_s1.py
exit 0
