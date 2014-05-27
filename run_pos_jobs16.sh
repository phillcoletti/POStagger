#!/bin/bash
# The name of the job, can be anything, simply used when displaying the list of running jobs
#$ -N L_w1_s1.py
# Combining output/error messages into one file
#$ -j y
# Set memory request:
#$ -l vf=2G
# Set walltime request:
#$ -l h_rt=23:00:00
# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username
#$ -cwd
# then you tell it retain all environment variables (as the default is to scrub your environment)
#$ -V
# Now comes the command to be executed
module load python
python evalTaggerW_w3.py 3
exit 0
