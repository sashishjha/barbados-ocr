#!/bin/bash
#SBATCH --job-name=r13_vlm_trio
#SBATCH --partition=debug
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/r13_trio_%j.out
#SBATCH --error=logs_checkpoint/r13_trio_%j.err

# R13 (2 Oct; deadline 4 Oct 23:59 UTC): add genuinely different ensemble members.
#   A1 Qwen2.5-VL-32B, fold 0 held out, 2 epochs -> val+test n-best   configs/qwen25vl_32b_r13.yaml
#   A2 chain: TrOCR-large all data (warm start from BEST_trocr_hw_r5, 6 epochs) -> test n-best,
#      then Qwen2.5-VL-7B all data seed 13 -> test n-best          configs/trocr_r13_all.yaml, qwen_recipe_7b_r12_all_s13.yaml
#   A3 Qwen2.5-VL-7B all data seed 17 -> test n-best              configs/qwen_recipe_7b_r13_all_s17.yaml
# No test pseudo-labels. VLM inference with 3 beams. Auto-retry from checkpoints. Node safety as in 26-32.

source slurm/common.sh
echo "== R13 launch on $(hostname) at $(date) =="
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
    ) > "logs_checkpoint/r13_${name}_${JOB_TAG}.log" 2>&1 &
    ARM_PIDS+=($!)
    echo "launched $name on allocated GPU $gpu (pid $!), log logs_checkpoint/r13_${name}_${JOB_TAG}.log"
}

# GPU B chain: TrOCR all data (~1 h) then Qwen2.5-7B all data seed 13 (~9 h), so the GPU never idles.
chain_b() {
    local gpu=$1
    (
      export HIP_VISIBLE_DEVICES=$gpu
      set -e
      ok=0
      for attempt in 1 2 3 4 5 6 7 8; do
          if python -m src.train_trocr --config configs/trocr_r13_all.yaml --resume \
                 --init-from runs_checkpoint/BEST_trocr_hw_r5_val0878 --set model.attn_implementation=eager; then ok=1; break; fi
          echo "== trocr_r13all: train attempt $attempt failed at $(date -Is); retrying in 120s =="; sleep 120
      done
      if [ "$ok" = 1 ]; then
          python -m src.infer --config configs/trocr_r13_all.yaml --checkpoint runs_checkpoint/trocr_r13all_checkpoints/best_checkpoint \
              --arm trocr --manifest data/Test.csv --nbest 5 --set model.attn_implementation=eager \
              --out "preds_checkpoint/trocr_r13all_test_nbest_${JOB_TAG}.csv" \
              --top1_out "preds_checkpoint/trocr_r13all_test_top1_${JOB_TAG}.csv" && echo "ARM trocr_r13all DONE"
      else
          echo "== trocr_r13all gave up; continuing with the Qwen2.5-7B seed 13 arm =="
      fi
    ) > "logs_checkpoint/r13_trocr_r13all_${JOB_TAG}.log" 2>&1
    run_arm qwen7b_r13all_s13 "$gpu" configs/qwen_recipe_7b_r12_all_s13.yaml "" "test"
    local st=0
    wait "${ARM_PIDS[-1]}" || st=$?
    return $st
}

run_arm qwen25vl32b_r13 "${ORDER[0]}" configs/qwen25vl_32b_r13.yaml "" "val test"
run_arm qwen7b_r13all_s17 "${ORDER[2]}" configs/qwen_recipe_7b_r13_all_s17.yaml "" "test"
chain_b "${ORDER[1]}" &
ARM_PIDS+=($!)
echo "launched chain trocr_r13all -> qwen7b_r13all_s13 on allocated GPU ${ORDER[1]} (pid $!)"

FAILED=0
for p in "${ARM_PIDS[@]}"; do
    wait "$p" || FAILED=$((FAILED + 1))
done
echo "== R13 done at $(date): $FAILED of ${#ARM_PIDS[@]} arms failed =="
exit $FAILED
