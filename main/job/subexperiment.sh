#!/bin/sh

qsub -l hostname=node414 run_experiment1_1.sh
qsub -l hostname=node413 run_experiment1_2.sh
qsub -l hostname=node416 run_experiment1_3.sh
qsub -l hostname=node415 run_experiment1_4.sh

