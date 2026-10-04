#!/bin/bash
#SBATCH --job-name=multiview
#SBATCH --partition=debug
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs_checkpoint/multiview_%j.out
#SBATCH --error=logs_checkpoint/multiview_%j.err
# Multi-view likelihood test (3 Oct): score every fold-0 candidate under K image views (height / vertical padding)
# with the same fold-0 models; if the view-averaged log-prob picks better candidates than the single training view,
# the same recipe is run on test. Outputs preds_checkpoint/lpv_val_<model>_<view>.csv
source slurm/common.sh
P3="$PWD/.pylib_qwen3_checkpoint"
declare -A VIEW=( [h224]="--height 224 --pad 0.2" [h288]="--height 288 --pad 0.2" [p08]="--height 256 --pad 0.08" [p35]="--height 256 --pad 0.35" )
view() {   # view GPU NAME CONFIG ADAPTER VIEWNAME [PYLIB]
    local gpu=$1 name=$2 cfg=$3 ad=$4 v=$5 pl=${6:-}
    local out=preds_checkpoint/lpv_val_${name}_${v}.csv log=logs_checkpoint/multiview_${name}_${v}.log
    grep -q "RESCORE DONE" $log 2>/dev/null && return 0
    for attempt in 1 2 3; do
        HIP_VISIBLE_DEVICES=$gpu PYTHONPATH="${pl:+$pl:}$PWD:$PYTHONPATH" python -m scripts.rescore_candidates \
            --config $cfg --adapter runs_checkpoint/$ad --cands preds_checkpoint/cands_val.csv --out $out ${VIEW[$v]} >> $log 2>&1 && return 0
        echo "$(date -Is) $name $v attempt $attempt failed" >> $log; sleep 60
    done
}
PIDS=()
stop_lanes() { echo "--- signal at $(date -Is) ---"; kill -TERM "${PIDS[@]}" 2>/dev/null; pkill -TERM -P $$ 2>/dev/null; exit 143; }
trap stop_lanes TERM INT
( for v in h224 h288 p08 p35; do view 0 q3_r8  configs/qwen3vl_8b.yaml        BEST_qwen3vl8b_r8_f55_08997 $v $P3; done ) & PIDS+=($!)
( for v in h224 h288 p08 p35; do view 1 q7_r8a configs/qwen_recipe_7b.yaml     BEST_qwen7b_r8a_f55_08986   $v;     done ) & PIDS+=($!)
( for v in h224 h288;         do view 2 q32    configs/qwen3vl_32b_r10.yaml    BEST_qwen3vl32b_r10_f55_09002 $v $P3; done ) & PIDS+=($!)
ST=0; for p in "${PIDS[@]}"; do wait "$p" || ST=$?; done
echo "== multiview done at $(date) (status $ST) =="
