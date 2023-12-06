import os

SLURM_OUTPUT_DIR = f"J{os.getenv('SLURM_JOB_ID')}_{os.getenv('SLURM_JOB_NAME')}"
