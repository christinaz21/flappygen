#!/bin/bash

WORKDIR=tmp/job_${RANDOM}
mkdir -p $WORKDIR
cp -r train_oasis $WORKDIR
cp -r config $WORKDIR

sbatch --export=WORKDIR="$WORKDIR" train.sh  # this export may not be safe
echo "WORKDIR: $WORKDIR"


# bash launch.sh