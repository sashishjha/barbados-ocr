#!/bin/bash
# Reuse GPU 0 of job 1603 (R12) after its resume arm finished:
# train Qwen3-VL-8B on ALL labelled data, seed 17, as an overlap step inside our own allocation,
# then test n-best. Checkpoints every 25 steps, so if job 1603 ends first, a later job can resume it from
# runs_checkpoint/qwen3vl8b_r12all_s17_checkpoints with the same config.
#   nohup setsid bash slurm/34_r12_gpu0_q3all_s17.sh > logs_checkpoint/r12_gpu0_launch_1603.out 2>&1 &
set -u
JOB=1603
GPU=0
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=logs_checkpoint/r12_qwen3vl8b_r12all_s17_${JOB}.log
echo "== start of qwen3vl8b_r12all_s17 on GPU $GPU at $(date -Is) ==" >> "$LOG"
exec srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=16 --mem=64G bash -c '
    source slurm/common.sh > /dev/null
    export HIP_VISIBLE_DEVICES='"$GPU"'
    export PYTHONPATH="$PWD/.pylib_qwen3_checkpoint:$PYTHONPATH"
    cfg=configs/qwen3vl_8b_r12_all_s17.yaml
    ckpt=runs_checkpoint/qwen3vl8b_r12all_s17_checkpoints
    trained=0
    for attempt in 1 2 3 4 5 6 7 8; do
        if python -m src.train_vlm --config "$cfg" --resume --set; then trained=1; break; fi
        echo "== qwen3vl8b_r12all_s17: train attempt $attempt failed at $(date -Is); retrying from latest checkpoint in 120s =="
        sleep 120
    done
    [ "$trained" = 1 ] || exit 1
    python -m src.infer --config "$cfg" --checkpoint "$ckpt/best_checkpoint" --arm vlm \
        --manifest data/Test.csv --nbest 3 --set generate.num_beams=3 \
        --out "preds_checkpoint/qwen3vl8b_r12all_s17_test_nbest_'"$JOB"'.csv" \
        --top1_out "preds_checkpoint/qwen3vl8b_r12all_s17_test_top1_'"$JOB"'.csv" || exit 1
    echo "ARM qwen3vl8b_r12all_s17 DONE"
' >> "$LOG" 2>&1
