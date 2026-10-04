#!/bin/bash
#SBATCH --job-name=r8_vlm_trio
#SBATCH --partition=debug
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/r8_trio_%j.out
#SBATCH --error=logs_checkpoint/r8_trio_%j.err

# R8: three VLM arms in parallel, one per allocated GPU, all with the public 0.904 recipe
# (bf16, LoRA r32 all-linear, 256-px lines) and 0.5/0.5 checkpoint selection:
#   A1 Qwen2.5-VL-7B  + R5 pseudo-labels          configs/qwen_recipe_7b.yaml
#   A2 Qwen3-VL-8B    + R5 pseudo-labels          configs/qwen3vl_8b.yaml   (transformers 4.57.1)
#   A3 Qwen2.5-VL-7B  variant: seed 7, lr 5e-5, StackMix, no pseudo-labels (ensemble diversity)
# Each arm: train -> n-best inference on fold-0 val (for ensemble weights) and on Test.
# Node safety as in 25_train_r7.sbatch: wait on every arm, forward TERM, VRAM preflight.

source slurm/common.sh
echo "== R8 launch on $(hostname) at $(date) =="
echo "SLURM GPUs: ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
JOB_TAG="${SLURM_JOB_ID:-$$}"
PSEUDO_CSV="data/derived_checkpoint/train_and_pseudo_r5.csv"
VAL_MANIFEST="data/derived_checkpoint/val_fold0_ids.csv"
MIN_FREE_GB="${MIN_FREE_GB:-60}"
mkdir -p logs_checkpoint preds_checkpoint

fail() { echo "FATAL: $1" >&2; exit 1; }
[ -f "$PSEUDO_CSV" ] || fail "missing $PSEUDO_CSV"
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

# run_arm NAME GPU CONFIG PYLIB EXTRA_SET...
run_arm() {
    local name=$1 gpu=$2 cfg=$3 pylib=$4; shift 4
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
      for split in val test; do
          man=$([ $split = val ] && echo "$VAL_MANIFEST" || echo data/Test.csv)
          python -m src.infer --config "$cfg" --checkpoint "$ckpt/best_checkpoint" --arm vlm \
              --manifest "$man" --nbest 5 \
              --out "preds_checkpoint/${name}_${split}_nbest_${JOB_TAG}.csv" \
              --top1_out "preds_checkpoint/${name}_${split}_top1_${JOB_TAG}.csv"
      done
      echo "ARM $name DONE"
    ) > "logs_checkpoint/r8_${name}_${JOB_TAG}.log" 2>&1 &
    ARM_PIDS+=($!)
    echo "launched $name on allocated GPU $gpu (pid $!), log logs_checkpoint/r8_${name}_${JOB_TAG}.log"
}

run_arm qwen7b_a  "${ORDER[0]}" configs/qwen_recipe_7b.yaml   ""                        data.pseudo_csv=$PSEUDO_CSV data.pseudo_loss_weight=1.0
run_arm qwen3vl8b "${ORDER[1]}" configs/qwen3vl_8b.yaml       .pylib_qwen3_checkpoint   data.pseudo_csv=$PSEUDO_CSV data.pseudo_loss_weight=1.0
run_arm qwen7b_b  "${ORDER[2]}" configs/qwen_recipe_7b_b.yaml ""                        data.pseudo_csv=null

FAILED=0
for p in "${ARM_PIDS[@]}"; do
    wait "$p" || FAILED=$((FAILED + 1))
done
echo "== R8 done at $(date): $FAILED of ${#ARM_PIDS[@]} arms failed =="
exit $FAILED
