#!/bin/bash
#SBATCH --job-name=r18_vlm_trio
#SBATCH --partition=debug
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/r18_trio_%j.out
#SBATCH --error=logs_checkpoint/r18_trio_%j.err

# R18 (4 Oct 05:10): upgrade MORE ensemble members with the vision-LoRA r128 mechanism (external advice + our finding that one upgraded member
# cannot flip a plurality vote).  Arms resume from head-start checkpoints:
#   r18_q7_vis128       Qwen2.5-VL-7B, fold 0 (validates transfer), val each epoch, then val+test inference
#   r18_q7_vis128_all   Qwen2.5-VL-7B ALL data, then test inference
#   r18_q3_vis128_all_s7  Qwen3-VL-8B vis128 ALL data, seed 7, then test inference
# The script waits for every arm (allocation stays alive).

source slurm/common.sh
echo "== R18 launch on $(hostname) at $(date) =="
echo "SLURM GPUs: ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
JOB_TAG="${SLURM_JOB_ID:-$$}"
touch "logs_checkpoint/KEEP_ALIVE_${JOB_TAG}"   # removed by hand when no overlap steps remain
VAL_MANIFEST="data/derived_checkpoint/val_fold0_ids.csv"
MIN_FREE_GB="${MIN_FREE_GB:-60}"
mkdir -p logs_checkpoint preds_checkpoint

fail() { echo "FATAL: $1" >&2; exit 1; }
[ -d .pylib_qwen3_checkpoint ] || fail "missing .pylib_qwen3_checkpoint (transformers 4.57.1 for Qwen3-VL)"
python -c "import pandas as pd; d=pd.read_csv('data/derived_checkpoint/train_folds.csv'); d[d.fold==0][['ID']].to_csv('$VAL_MANIFEST', index=False)" \
    || fail "could not write $VAL_MANIFEST"

# --- VRAM preflight: rank allocated GPUs by free memory (other services share the node) ---
declare -a FREE ORDER
for idx in 0 1 2; do
    FREE[$idx]=$(HIP_VISIBLE_DEVICES=$idx python -c \
        "import torch; f,_=torch.cuda.mem_get_info(0); print(int(f/2**30))") \
        || fail "could not query allocated GPU $idx"
    echo "allocated GPU $idx: ${FREE[$idx]} GB free"
    [ "${FREE[$idx]}" -ge "$MIN_FREE_GB" ] || fail "allocated GPU $idx has only ${FREE[$idx]} GB free (< $MIN_FREE_GB)"
done
read -r -a ORDER <<< "$(for i in 0 1 2; do echo "${FREE[$i]} $i"; done | sort -rn | awk '{printf "%s ", $2}')"

ARM_PIDS=()
stop_arms() {
    echo "--- signal at $(date -Is): stopping arms ${ARM_PIDS[*]} ---"
    kill -TERM "${ARM_PIDS[@]}" 2>/dev/null || true
    wait "${ARM_PIDS[@]}" 2>/dev/null || true
    exit 143
}
trap stop_arms TERM INT

# run_arm NAME GPU CONFIG PYLIB SPLITS EXTRA_SET...   (SPLITS: "val test" or "test")
run_arm() {
    local name=$1 gpu=$2 cfg=$3 pylib=$4 splits=$5; shift 5
    local ckpt; ckpt=$(python -c "from src.utils.common import load_config as l; print(l('$cfg')['output']['checkpoint_dir'])")
    (
      export HIP_VISIBLE_DEVICES=$gpu
      [ -n "$pylib" ] && export PYTHONPATH="$PWD/$pylib:$PYTHONPATH"
      set -e
      # Auto-retry: shared GPUs can OOM at any time. Each retry resumes from the latest
      # atomic step-checkpoint (<= ~25 steps lost); wait between tries for memory to free up.
      trained=0
      for attempt in 1 2 3 4 5 6 7 8; do
          if python -m src.train_vlm --config "$cfg" --resume --set "$@"; then trained=1; break; fi
          echo "== $name: train attempt $attempt failed at $(date -Is); retrying from latest checkpoint in 120s =="
          sleep 120
      done
      [ "$trained" = 1 ]
      for split in $splits; do
          man=$([ $split = val ] && echo "$VAL_MANIFEST" || echo data/Test.csv)
          python -m src.infer --config "$cfg" --checkpoint "$ckpt/best_checkpoint" --arm vlm \
              --manifest "$man" --nbest 3 --set generate.num_beams=3 \
              --out "preds_checkpoint/${name}_${split}_nbest_${JOB_TAG}.csv" \
              --top1_out "preds_checkpoint/${name}_${split}_top1_${JOB_TAG}.csv"
      done
      echo "ARM $name DONE"
    ) > "logs_checkpoint/r18_${name}_${JOB_TAG}.log" 2>&1 &
    ARM_PIDS+=($!)
    echo "launched $name on allocated GPU $gpu (pid $!), log logs_checkpoint/r18_${name}_${JOB_TAG}.log"
}

run_arm r18_q7_vis128        "${ORDER[0]}" configs/qwen_recipe_7b_r18_vis128.yaml      ""                       "val test"
run_arm r18_q7_vis128_all    "${ORDER[1]}" configs/qwen_recipe_7b_r18_vis128_all.yaml  ""                       "test"
run_arm r18_q3_vis128_all_s7 "${ORDER[2]}" configs/qwen3vl_8b_r18_vis128_all_s7.yaml   .pylib_qwen3_checkpoint  "test"

FAILED=0
for p in "${ARM_PIDS[@]}"; do
    wait "$p" || FAILED=$((FAILED + 1))
done
# KEEP-ALIVE GUARD (lesson of 4 Oct 01:28): extra runs may be attached to this allocation as overlap steps; the allocation
# must not be released while they run.  Delete the sentinel file to let the job end.
while [ -e "logs_checkpoint/KEEP_ALIVE_${JOB_TAG}" ]; do sleep 60; done
echo "== R18 done at $(date): $FAILED of ${#ARM_PIDS[@]} arms failed =="
exit $FAILED
