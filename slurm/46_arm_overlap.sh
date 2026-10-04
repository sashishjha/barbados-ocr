#!/bin/bash
# Run one arm as an overlap step inside job JOB on allocated GPU idx GPU:  [train with retries] then [inference val/test].
#   bash slurm/46_arm_overlap.sh JOB GPU NAME CONFIG "SPLITS" MODE      MODE = train+infer | infer
# Outputs: preds_checkpoint/<NAME>_{val,test}_nbest_<JOB>.csv (3 beams, n-best 3); log logs_checkpoint/auto_<NAME>_<JOB>.log
set -u
JOB=$1; GPU=$2; NAME=$3; CFG=$4; SPLITS=$5; MODE=${6:-train+infer}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=logs_checkpoint/auto_${NAME}_${JOB}.log
echo "== $NAME ($MODE) on GPU $GPU at $(date -Is) cfg=$CFG splits='$SPLITS' ==" >> "$LOG"
srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=16 --mem=64G bash -c '
    source slurm/common.sh > /dev/null
    export HIP_VISIBLE_DEVICES='"$GPU"' PYTHONPATH="$PWD/.pylib_qwen3_checkpoint:$PYTHONPATH"
    CKPT=$(python -c "from src.utils.common import load_config as l; print(l(\"'"$CFG"'\")[\"output\"][\"checkpoint_dir\"])" | tail -1)
    if [ "'"$MODE"'" = "train+infer" ]; then
        ok=0
        for attempt in 1 2 3 4 5 6; do
            python -m src.train_vlm --config '"$CFG"' --resume --set && { ok=1; break; }
            echo "== '"$NAME"': train attempt $attempt failed at $(date -Is); retry in 120s =="; sleep 120
        done
        [ $ok = 1 ] || { echo "TRAIN FAILED"; exit 1; }
        echo "TRAIN FINISHED"
    fi
    for split in '"$SPLITS"'; do
        man=$([ $split = val ] && echo data/derived_checkpoint/val_fold0_ids.csv || echo data/Test.csv)
        python -m src.infer --config '"$CFG"' --checkpoint $CKPT/best_checkpoint --arm vlm --manifest $man --nbest 3 --set generate.num_beams=3 \
            --out preds_checkpoint/'"$NAME"'_${split}_nbest_'"$JOB"'.csv --top1_out preds_checkpoint/'"$NAME"'_${split}_top1_'"$JOB"'.csv || { echo "INFER $split FAILED"; exit 2; }
    done
    echo "ARM '"$NAME"' ALL DONE"
' >> "$LOG" 2>&1
