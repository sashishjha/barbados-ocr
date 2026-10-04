#!/bin/bash
#SBATCH --job-name=r9_vlm_trio
#SBATCH --partition=debug
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/r9_trio_%j.out
#SBATCH --error=logs_checkpoint/r9_trio_%j.err

# R9: retrain the two best R8 models on R9 pseudo-labels (fused R8 ensemble, LB 0.9154,
# kept only where the arms agree), fold 0 held out, validation only after the last epoch:
#   A1 Qwen3-VL-8B               configs/qwen3vl_8b_r9.yaml     (transformers 4.57.1)
#   A2 Qwen2.5-VL-7B new recipe  configs/qwen_recipe_7b_r9.yaml
#   A3 TrOCR R7 val+test n-best (missing since job 1358 was cancelled), then
#      Qwen3-VL-8B seed 7         configs/qwen3vl_8b_r9_s7.yaml  (ensemble diversity)
# Each arm: train (auto-retry, resumes from 25-step checkpoints) -> n-best on fold-0 val + Test.
# Node safety as in 26_train_vlm_trio.sbatch: wait on every arm, forward TERM, VRAM preflight.

source slurm/common.sh
echo "== R9 launch on $(hostname) at $(date) =="
echo "SLURM GPUs: ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
JOB_TAG="${SLURM_JOB_ID:-$$}"
PSEUDO_CSV="data/derived_checkpoint/pseudo_r9.csv"
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

# run_arm NAME GPU CONFIG PYLIB PRE EXTRA_SET...   (PRE: shell function run first on that GPU, or "")
run_arm() {
    local name=$1 gpu=$2 cfg=$3 pylib=$4 pre=$5; shift 5
    local ckpt; ckpt=$(python -c "from src.utils.common import load_config as l; print(l('$cfg')['output']['checkpoint_dir'])")
    (
      export HIP_VISIBLE_DEVICES=$gpu
      # pre-step runs on the default transformers (TrOCR R7 was trained on it), before pylib is added
      if [ -n "$pre" ]; then $pre || echo "== $name: pre-step $pre FAILED (continuing with training) =="; fi
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
    ) > "logs_checkpoint/r9_${name}_${JOB_TAG}.log" 2>&1 &
    ARM_PIDS+=($!)
    echo "launched $name on allocated GPU $gpu (pid $!), log logs_checkpoint/r9_${name}_${JOB_TAG}.log"
}

trocr_r7_infer() {
    for split in val test; do
        man=$([ $split = val ] && echo "$VAL_MANIFEST" || echo data/Test.csv)
        python -m src.infer --config configs/trocr_large.yaml --checkpoint runs_checkpoint/BEST_trocr_hw_r7_val0887 \
            --arm trocr --manifest "$man" --nbest 5 --set model.attn_implementation=eager \
            --out "preds_checkpoint/trocr_r7_${split}_nbest_${JOB_TAG}.csv" \
            --top1_out "preds_checkpoint/trocr_r7_${split}_top1_${JOB_TAG}.csv" || return 1
    done
    echo "TrOCR R7 inference DONE"
}

run_arm qwen3vl8b_r9   "${ORDER[0]}" configs/qwen3vl_8b_r9.yaml    .pylib_qwen3_checkpoint ""             data.pseudo_csv=$PSEUDO_CSV
run_arm qwen7b_r9a     "${ORDER[1]}" configs/qwen_recipe_7b_r9.yaml ""                     ""             data.pseudo_csv=$PSEUDO_CSV
run_arm qwen3vl8b_r9s7 "${ORDER[2]}" configs/qwen3vl_8b_r9_s7.yaml .pylib_qwen3_checkpoint trocr_r7_infer data.pseudo_csv=$PSEUDO_CSV

FAILED=0
for p in "${ARM_PIDS[@]}"; do
    wait "$p" || FAILED=$((FAILED + 1))
done
echo "== R9 done at $(date): $FAILED of ${#ARM_PIDS[@]} arms failed =="
exit $FAILED
