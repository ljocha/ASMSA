export KERASTUNER_TUNER_ID="chief"
export KERASTUNER_ORACLE_IP="0.0.0.0"
export KERASTUNER_ORACLE_PORT="8999"
export OMP_NUM_THREADS=1
python3 tuning.py "$@"
