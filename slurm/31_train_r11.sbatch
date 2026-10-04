#!/bin/bash
#SBATCH --job-name=r11_exp_trio
#SBATCH --partition=debug
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/r11_trio_%j.out
#SBATCH --error=logs_checkpoint/r11_trio_%j.err

# R11 (1 Oct, plan made after 32B was no better alone but +0.0044 as an ensemble member):
#   A1 Qwen3-VL-32B on ALL labelled data, 2 epochs -> test n-best      configs/qwen3vl_32b_r11_all.yaml
#   A2 Qwen3-VL-8B at 384-px lines, fold 0 held out -> val+test n-best configs/qwen3vl_8b_r11_res384.yaml
#   A3 inference only: test-time augmentation + beam check on the banked Qwen3-VL-8B R8 model
#      (fold 0 held out, val 0.8997 at 256 px / 5 beams): heights 224 and 320, and 3 beams;
#      all val variants first (decision data early), then test.
# Training arms: auto-retry from 25-step checkpoints. Node safety as in 26-29.

source slurm/common.sh
echo "== R11 launch on $(hostname) at $(date) =="
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
              --manifest "$man" --nbest 5 \
              --out "preds_checkpoint/${name}_${split}_nbest_${JOB_TAG}.csv" \
              --top1_out "preds_checkpoint/${name}_${split}_top1_${JOB_TAG}.csv"
      done
      echo "ARM $name DONE"
    ) > "logs_checkpoint/r11_${name}_${JOB_TAG}.log" 2>&1 &
    ARM_PIDS+=($!)
    echo "launched $name on allocated GPU $gpu (pid $!), log logs_checkpoint/r11_${name}_${JOB_TAG}.log"
}

# tta_arm GPU: inference-only variants of BEST_qwen3vl8b_r8 (no training)
tta_arm() {
    local gpu=$1
    (
      export HIP_VISIBLE_DEVICES=$gpu
      export PYTHONPATH="$PWD/.pylib_qwen3_checkpoint:$PYTHONPATH"
      ck=runs_checkpoint/BEST_qwen3vl8b_r8_f55_08997
      for split in val test; do
          man=$([ $split = val ] && echo "$VAL_MANIFEST" || echo data/Test.csv)
          for v in "h224|5|model.image_height=224 model.min_pixels=175616" \
                   "h320|5|model.image_height=320 model.image_max_width=5120 model.max_pixels=2007040" \
                   "b3|3|generate.num_beams=3"; do
              IFS="|" read -r tag nb sets <<< "$v"
              echo "== TTA $tag $split start $(date -Is) =="
              python -m src.infer --config configs/qwen3vl_8b.yaml --checkpoint "$ck" --arm vlm \
                  --manifest "$man" --nbest "$nb" --set $sets \
                  --out "preds_checkpoint/tta_q3r8_${tag}_${split}_nbest_${JOB_TAG}.csv" \
                  --top1_out "preds_checkpoint/tta_q3r8_${tag}_${split}_top1_${JOB_TAG}.csv" \
                  || echo "== TTA $tag $split FAILED =="
          done
      done
      echo "ARM tta_q3r8 DONE"
    ) > "logs_checkpoint/r11_tta_q3r8_${JOB_TAG}.log" 2>&1 &
    ARM_PIDS+=($!)
    echo "launched tta_q3r8 on allocated GPU $gpu (pid $!), log logs_checkpoint/r11_tta_q3r8_${JOB_TAG}.log"
}

run_arm qwen3vl32b_r11all    "${ORDER[0]}" configs/qwen3vl_32b_r11_all.yaml   .pylib_qwen3_checkpoint "test"
run_arm qwen3vl8b_r11_res384 "${ORDER[1]}" configs/qwen3vl_8b_r11_res384.yaml .pylib_qwen3_checkpoint "val test"
tta_arm "${ORDER[2]}"

FAILED=0
for p in "${ARM_PIDS[@]}"; do
    wait "$p" || FAILED=$((FAILED + 1))
done
echo "== R11 done at $(date): $FAILED of ${#ARM_PIDS[@]} arms failed =="
exit $FAILED
