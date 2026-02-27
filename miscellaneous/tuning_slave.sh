#!/bin/bash

echo \#threads: ${OMP_NUM_THREADS:=4}
export OMP_NUM_THREADS
export KERASTUNER_TUNER_ID="$1"
export KERASTUNER_ORACLE_IP=$(echo $2 | cut -d: -f1)
export KERASTUNER_ORACLE_PORT=$(echo $2 | cut -d: -f2)

shift
shift
python3 tuning.py "$@"
