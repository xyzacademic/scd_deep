#!/bin/sh

for ((i=1;i<=10;i++))
do
    python generate_job.py $i
done