#!/bin/sh
#$ -V 
#$ -cwd 
#$ -o world 
#$ -N hello 
#$ -M yabdelrahman1999@gmail.com
#$ -m e
# echo STARTED at $(date)
# echo “Hello World”
# # Used resources
# qstat -j $JOB_ID | awk 'NR==1,/^scheduling info:/'
# echo FINISHED at $(date)  
# python pyscript.py
source /nethome/2514818/miniconda3/etc/profile.d/conda.sh
conda activate py310
python code_yassin/RadioTrain.py
# HOW TO RUN:
# qsub -q long.q /nethome/2514818/X-3D/code_yassin/world.sh
