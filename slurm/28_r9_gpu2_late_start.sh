#!/bin/bash
# Late start of the R9 seed-7 arm (qwen3vl8b_r9s7) on allocated GPU 2 of job 1430.
# The arm exhausted its retries at 10:29 on 30 Sep because a non-SLURM vLLM server holds
# ~162 GB of that GPU. This loop runs on the login node (no GPU use), polls GPU 2 every
# 5 min, and once >= NEED_GB is free launches the arm as an overlap step INSIDE job 1430
# (our own allocation). It gives up at DEADLINE: the arm needs ~9 h and is killed when
# job 1430 ends (after the two main arms finish, ~00:30).
#   nohup setsid bash slurm/28_r9_gpu2_late_start.sh > logs_checkpoint/r9_gpu2_late_1430.out 2>&1 &
set -u
JOB=1430
GPU=2
NEED_GB=100
DEADLINE="${DEADLINE:-14:30}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=logs_checkpoint/r9_qwen3vl8b_r9s7_${JOB}.log
end=$(date -d "today $DEADLINE" +%s)

gpu_free_gb() {
    rocm-smi --showmeminfo vram 2>/dev/null | awk -v g="GPU[$GPU]" '
        index($0, g) == 1 && /Total Memory/ {t=$NF} index($0, g) == 1 && /Total Used/ {u=$NF}
        END {printf "%d", (t-u)/2^30}'
}

while :; do
    if ! squeue -h -j $JOB -o %T | grep -q RUNNING; then
        echo "$(date -Is) job $JOB not running any more; exiting"; exit 0
    fi
    free=$(gpu_free_gb)
    echo "$(date -Is) GPU $GPU free: ${free} GB (need $NEED_GB)"
    if [ "${free:-0}" -ge $NEED_GB ]; then
        echo "== late start of qwen3vl8b_r9s7 on GPU $GPU at $(date -Is) (${free} GB free) ==" >> "$LOG"
        exec srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=16 --mem=64G bash -c '
            source slurm/common.sh > /dev/null
            export HIP_VISIBLE_DEVICES='"$GPU"'
            export PYTHONPATH="$PWD/.pylib_qwen3_checkpoint:$PYTHONPATH"
            cfg=configs/qwen3vl_8b_r9_s7.yaml
            ckpt=runs_checkpoint/qwen3vl8b_r9s7_checkpoints
            trained=0
            for attempt in 1 2 3 4 5 6 7 8; do
                if python -m src.train_vlm --config "$cfg" --resume --set data.pseudo_csv=data/derived_checkpoint/pseudo_r9.csv; then
                    trained=1; break
                fi
                echo "== qwen3vl8b_r9s7: train attempt $attempt failed at $(date -Is); retrying from latest checkpoint in 120s =="
                sleep 120
            done
            [ "$trained" = 1 ] || exit 1
            for split in val test; do
                man=$([ $split = val ] && echo data/derived_checkpoint/val_fold0_ids.csv || echo data/Test.csv)
                python -m src.infer --config "$cfg" --checkpoint "$ckpt/best_checkpoint" --arm vlm \
                    --manifest "$man" --nbest 5 \
                    --out "preds_checkpoint/qwen3vl8b_r9s7_${split}_nbest_'"$JOB"'.csv" \
                    --top1_out "preds_checkpoint/qwen3vl8b_r9s7_${split}_top1_'"$JOB"'.csv" || exit 1
            done
            echo "ARM qwen3vl8b_r9s7 DONE"
        ' >> "$LOG" 2>&1
    fi
    if [ "$(date +%s)" -ge "$end" ]; then
        echo "$(date -Is) deadline $DEADLINE reached without $NEED_GB GB free on GPU $GPU; giving up"
        echo "== qwen3vl8b_r9s7 late start: GPU $GPU never freed before $DEADLINE; arm skipped ==" >> "$LOG"
        exit 0
    fi
    sleep 300
done
