## Commands

```commandline
# to submit to a queue
sbatch 

# to run interactive mode
srun

#inspect status of your jobs
squeue --long

# to abort job
scancel <JOBID>

# statistics about your job
sstat <JOBID>

# Provides detailed information about a specific job, including resource allocation, state, and node information.

scontrol show job 12345
scontrol show log <job_id>
```


