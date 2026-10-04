#!/bin/bash
# Head-start training on an idle GPU of a running job (overlap step inside our own allocation).
# Training only: the next job's arm with the SAME config resumes from these 25-step checkpoints
# (--resume) and then does its own inference, so nothing is wasted when this job ends.
#   nohup setsid bash slurm/36_overlap_pretrain.sh JOB GPU NAME CONFIG [PYLIB] > logs_checkpoint/pre_NAME.out 2>&1 &
set -u
JOB=$1; GPU=$2; NAME=$3; CFG=$4; PYLIB=${5:-}; MEM=${MEM:-64G}   # MEM=120G for 32B models (CPU RAM while loading 66 GB of weights)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=logs_checkpoint/pre_${NAME}_${JOB}.log
echo "== head-start training of $NAME on GPU $GPU (job $JOB) at $(date -Is) ==" >> "$LOG"
exec srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=16 --mem=$MEM bash -c '
    source slurm/common.sh > /dev/null
    export HIP_VISIBLE_DEVICES='"$GPU"'
    [ -n "'"$PYLIB"'" ] && export PYTHONPATH="$PWD/'"$PYLIB"':$PYTHONPATH"
    for attempt in 1 2 3; do
        python -m src.train_vlm --config '"$CFG"' --resume --set && { echo "PRETRAIN '"$NAME"' FINISHED"; exit 0; }
        echo "== '"$NAME"': attempt $attempt failed at $(date -Is); retrying in 120s =="; sleep 120
    done
' >> "$LOG" 2>&1
