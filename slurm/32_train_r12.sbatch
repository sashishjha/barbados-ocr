#!/bin/bash
#SBATCH --job-name=r12_vlm_trio
#SBATCH --partition=debug
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/r12_trio_%j.out
#SBATCH --error=logs_checkpoint/r12_trio_%j.err

# R12: final training round (2 Oct). 384-px lines gave val 0.9001 vs 0.8987 (no ensemble gain) and an
# all-data 32B scored LB 0.9189 vs 0.9207, so R12 adds more ALL-data models of the two Qwen families:
#   A1 Qwen2.5-VL-7B all data, seed 7    configs/qwen_recipe_7b_r11_all_s7.yaml
#   A2 Qwen3-VL-8B  all data, seed 13    configs/qwen3vl_8b_r11_all_s13.yaml
#   A3 if R11's Qwen3-VL-8B all-data seed 7 never wrote its test n-best: resume it (same config, --resume);
#      else Qwen2.5-VL-7B all data, seed 13 (configs/qwen_recipe_7b_r12_all_s13.yaml)
# No test pseudo-labels, no validation; test n-best with 3 beams (= 5 beams in accuracy, faster).
# Auto-retry from 25-step checkpoints. Node safety as in 26-31.

source slurm/common.sh
echo "== R12 launch on $(hostname) at $(date) =="
echo "SLURM GPUs: ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
JOB_TAG="${SLURM_JOB_ID:-$$}"
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
    ) > "logs_checkpoint/r12_${name}_${JOB_TAG}.log" 2>&1 &
    ARM_PIDS+=($!)
    echo "launched $name on allocated GPU $gpu (pid $!), log logs_checkpoint/r12_${name}_${JOB_TAG}.log"
}

run_arm qwen7b_r12all_s7     "${ORDER[0]}" configs/qwen_recipe_7b_r11_all_s7.yaml ""                      "test"
run_arm qwen3vl8b_r12all_s13 "${ORDER[1]}" configs/qwen3vl_8b_r11_all_s13.yaml  .pylib_qwen3_checkpoint "test"
if [ -s preds_checkpoint/qwen3vl8b_r11all_s7_test_nbest_1519.csv ]; then
    run_arm qwen7b_r12all_s13 "${ORDER[2]}" configs/qwen_recipe_7b_r12_all_s13.yaml "" "test"
else
    echo "R11 qwen3vl8b_r11all_s7 unfinished -> resuming it on GPU ${ORDER[2]}"
    run_arm qwen3vl8b_r11all_s7 "${ORDER[2]}" configs/qwen3vl_8b_r11_all_s7.yaml .pylib_qwen3_checkpoint "test"
fi

FAILED=0
for p in "${ARM_PIDS[@]}"; do
    wait "$p" || FAILED=$((FAILED + 1))
done
echo "== R12 done at $(date): $FAILED of ${#ARM_PIDS[@]} arms failed =="
exit $FAILED
