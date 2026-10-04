#!/bin/bash
# Reuse GPU 1 of job 1519 (R11) after its TTA arm was stopped (TTA gave no ensemble gain):
# train Qwen3-VL-8B on ALL labelled data, seed 7, as an overlap step inside our own allocation,
# then test n-best. Checkpoints every 25 steps, so if job 1519 ends first, R12 resumes it from
# runs_checkpoint/qwen3vl8b_r11all_s7_checkpoints with the same config.
#   nohup setsid bash slurm/33_r11_gpu1_q3all_s7.sh > logs_checkpoint/r11_gpu1_launch_1519.out 2>&1 &
set -u
JOB=1519
GPU=1
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=logs_checkpoint/r11_qwen3vl8b_r11all_s7_${JOB}.log
echo "== start of qwen3vl8b_r11all_s7 on GPU $GPU at $(date -Is) ==" >> "$LOG"
exec srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=16 --mem=64G bash -c '
    source slurm/common.sh > /dev/null
    export HIP_VISIBLE_DEVICES='"$GPU"'
    export PYTHONPATH="$PWD/.pylib_qwen3_checkpoint:$PYTHONPATH"
    cfg=configs/qwen3vl_8b_r11_all_s7.yaml
    ckpt=runs_checkpoint/qwen3vl8b_r11all_s7_checkpoints
    trained=0
    for attempt in 1 2 3 4 5 6 7 8; do
        if python -m src.train_vlm --config "$cfg" --resume --set; then trained=1; break; fi
        echo "== qwen3vl8b_r11all_s7: train attempt $attempt failed at $(date -Is); retrying from latest checkpoint in 120s =="
        sleep 120
    done
    [ "$trained" = 1 ] || exit 1
    python -m src.infer --config "$cfg" --checkpoint "$ckpt/best_checkpoint" --arm vlm \
        --manifest data/Test.csv --nbest 3 --set generate.num_beams=3 \
        --out "preds_checkpoint/qwen3vl8b_r11all_s7_test_nbest_'"$JOB"'.csv" \
        --top1_out "preds_checkpoint/qwen3vl8b_r11all_s7_test_top1_'"$JOB"'.csv" || exit 1
    echo "ARM qwen3vl8b_r11all_s7 DONE"
' >> "$LOG" 2>&1
