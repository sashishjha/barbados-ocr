#!/bin/bash
#SBATCH --job-name=rescore
#SBATCH --partition=debug
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/rescore_%j.out
#SBATCH --error=logs_checkpoint/rescore_%j.err
# Candidate rescoring (3 Oct): every trained VLM scores every candidate reading of each line (teacher forcing);
# scripts/select_candidates.py then picks per line. Oracle on fold-0 val: best existing candidate = 0.946 vs ROVER 0.916.
# Val first (fold-0 models), then test (models of the 0.9221 recipe). 3 lanes, one per GPU; each step retried.
source slurm/common.sh
JOB=${SLURM_JOB_ID}
P3="$PWD/.pylib_qwen3_checkpoint"
step() {   # step GPU NAME CONFIG ADAPTER SPLIT [PYLIB]
    local gpu=$1 name=$2 cfg=$3 ad=$4 split=$5 pl=${6:-}
    local out=preds_checkpoint/lp_${split}_${name}.csv
    grep -q "RESCORE DONE" logs_checkpoint/rescore_${split}_${name}.log 2>/dev/null && return 0
    for attempt in 1 2 3; do
        HIP_VISIBLE_DEVICES=$gpu PYTHONPATH="${pl:+$pl:}$PWD:$PYTHONPATH" python -m scripts.rescore_candidates \
            --config $cfg --adapter runs_checkpoint/$ad --cands preds_checkpoint/cands_${split}.csv --out $out \
            >> logs_checkpoint/rescore_${split}_${name}.log 2>&1 && return 0
        echo "$(date -Is) $name $split attempt $attempt failed" >> logs_checkpoint/rescore_${split}_${name}.log; sleep 60
    done
}
PIDS=()
stop_lanes() { echo "--- signal at $(date -Is): stopping lanes ${PIDS[*]} ---"; kill -TERM "${PIDS[@]}" 2>/dev/null; pkill -TERM -P $$ 2>/dev/null; exit 143; }
trap stop_lanes TERM INT
wait_32b() { until grep -q "DOWNLOAD DONE" logs_checkpoint/dl_qwen3vl32b.log 2>/dev/null; do sleep 60; done; }
( step 0 q3_r8   configs/qwen3vl_8b.yaml          BEST_qwen3vl8b_r8_f55_08997   val  $P3
  step 0 q3_all  configs/qwen3vl_8b_r10_all.yaml  BEST_qwen3vl8b_r10all_all     test $P3
  wait_32b
  step 0 q32     configs/qwen3vl_32b_r10.yaml     BEST_qwen3vl32b_r10_f55_09002 val  $P3
  step 0 q32     configs/qwen3vl_32b_r10.yaml     BEST_qwen3vl32b_r10_f55_09002 test $P3 ) &
PIDS+=($!)
( step 1 q7_r8a  configs/qwen_recipe_7b.yaml      BEST_qwen7b_r8a_f55_08986     val
  step 1 q7_all  configs/qwen_recipe_7b_r10_all.yaml BEST_qwen7b_r10all_all     test
  step 1 q3_s7   configs/qwen3vl_8b_r11_all_s7.yaml BEST_qwen3vl8b_r11all_s7_all test $P3 ) &
PIDS+=($!)
( step 2 q7_r8b  configs/qwen_recipe_7b_b.yaml    BEST_qwen7b_r8b_f55_08829     val
  step 2 q7_r8b  configs/qwen_recipe_7b_b.yaml    BEST_qwen7b_r8b_f55_08829     test ) &
PIDS+=($!)
ST=0
for p in "${PIDS[@]}"; do wait "$p" || ST=$?; done
echo "== rescore done at $(date) =="
