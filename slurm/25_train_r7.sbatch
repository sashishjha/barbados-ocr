#!/bin/bash
#SBATCH --job-name=r7_train_parallel
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/r7_train_%j.out
#SBATCH --error=logs_checkpoint/r7_train_%j.err

# R7: warm-start Qwen2.5-VL LoRA (arm A) and TrOCR (arm B) from the golden R5
# weights and train on base + R5 pseudo-labels, one arm per allocated GPU.
#
# Node-safety rules (job 1351 drained iisc by breaking these):
#   * never exit while an arm is still running: common.sh sets -e, so a bare
#     `wait` on a crashed arm killed the script and orphaned the other arm on
#     its GPU -> "Kill task failed" -> node DRAIN.
#   * forward scancel/timeout signals to the arms and wait for them to exit.
#   * never override the GPUs SLURM allocated; pick among them with
#     HIP_VISIBLE_DEVICES (relative index inside SLURM's ROCR_VISIBLE_DEVICES).

source slurm/common.sh

echo "== R7 launch on $(hostname) at $(date) =="
echo "SLURM GPUs: ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

JOB_TAG="${SLURM_JOB_ID:-$$}"

# --- Paths -------------------------------------------------------------------
QWEN_R5_GOLD="runs_checkpoint/BEST_qwen7b_r5_val0910"
TROCR_R5_GOLD="runs_checkpoint/BEST_trocr_hw_r5_val0878"

QWEN_R7_CKPT_DIR="runs_checkpoint/qwen25vl_lora_checkpoints_r7b"   # _r7 holds a from-scratch start (29 Sep)
QWEN_R7_OUT_DIR="runs_checkpoint/qwen25vl_lora_r7"

TROCR_R7_CKPT_DIR="runs_checkpoint/trocr_hw_checkpoints_r7b"      # _r7 touched by a duplicate run (29 Sep)
TROCR_R7_OUT_DIR="runs_checkpoint/trocr_hw_r7"

PSEUDO_CSV="data/derived_checkpoint/train_and_pseudo_r5.csv"
EXPECTED_PSEUDO_ROWS=4960
QWEN_MIN_FREE_GB="${QWEN_MIN_FREE_GB:-70}"    # Qwen peaked ~60 GB (eval step) in R7 attempt 1351
TROCR_MIN_FREE_GB="${TROCR_MIN_FREE_GB:-40}"  # TrOCR-large full fine-tune needs far less

mkdir -p logs_checkpoint

fail() { echo "FATAL: $1" >&2; exit 1; }

# --- Preflight checks (all before any GPU work) ------------------------------
[ -d "$QWEN_R5_GOLD" ]  || fail "Golden Qwen R5 checkpoint not found at $QWEN_R5_GOLD"
[ -d "$TROCR_R5_GOLD" ] || fail "Golden TrOCR R5 checkpoint not found at $TROCR_R5_GOLD"
[ -f "$PSEUDO_CSV" ]    || fail "Pseudo-label CSV not found at $PSEUDO_CSV"

ROW_COUNT=$(($(wc -l < "$PSEUDO_CSV") - 1))
if [ "$ROW_COUNT" -lt "$EXPECTED_PSEUDO_ROWS" ]; then
    fail "$PSEUDO_CSV has only $ROW_COUNT data rows, expected >= $EXPECTED_PSEUDO_ROWS. Refusing to launch R7."
fi
echo "Pseudo-label CSV OK: $ROW_COUNT rows"

# The node is shared with services outside SLURM, so an allocated GPU may be
# partly full. Put Qwen (bigger) on the allocated GPU with more free memory and
# refuse to start if either arm would not fit, instead of OOM-ing mid-run.
declare -a FREE
for idx in 0 1; do
    FREE[$idx]=$(HIP_VISIBLE_DEVICES=$idx python -c \
        "import torch; f,_=torch.cuda.mem_get_info(0); print(int(f/2**30))") \
        || fail "could not query free memory on allocated GPU $idx"
    echo "allocated GPU $idx: ${FREE[$idx]} GB free"
done
if [ "${FREE[0]}" -ge "${FREE[1]}" ]; then QWEN_GPU=0; TROCR_GPU=1; else QWEN_GPU=1; TROCR_GPU=0; fi
[ "${FREE[$QWEN_GPU]}" -ge "$QWEN_MIN_FREE_GB" ] \
    || fail "best allocated GPU has only ${FREE[$QWEN_GPU]} GB free (< ${QWEN_MIN_FREE_GB} GB needed by Qwen)"
[ "${FREE[$TROCR_GPU]}" -ge "$TROCR_MIN_FREE_GB" ] \
    || fail "other allocated GPU has only ${FREE[$TROCR_GPU]} GB free (< ${TROCR_MIN_FREE_GB} GB needed by TrOCR)"

# Working dirs are NOT wiped: if this job is requeued or relaunched, --resume
# continues from its own checkpoint_step*; --init-from only applies to a fresh dir.
mkdir -p "$QWEN_R7_CKPT_DIR" "$TROCR_R7_CKPT_DIR" "$QWEN_R7_OUT_DIR" "$TROCR_R7_OUT_DIR"

# --- Signal handling: never leave an arm orphaned on a GPU -------------------
ARM_PIDS=()
stop_arms() {
    echo "--- signal received at $(date -Is): stopping arms ${ARM_PIDS[*]} ---"
    kill -TERM "${ARM_PIDS[@]}" 2>/dev/null || true
    wait "${ARM_PIDS[@]}" 2>/dev/null || true
    exit 143
}
trap stop_arms TERM INT

# --- Launch Qwen ------------------------------------------------------------------
echo "Launching Qwen2.5-VL-7B LoRA training on allocated GPU $QWEN_GPU..."
(
  export HIP_VISIBLE_DEVICES=$QWEN_GPU
  exec python -m src.train_vlm --config configs/qwen25vl_lora.yaml \
      --resume --init-from "$QWEN_R5_GOLD" \
      --set data.pseudo_csv="$PSEUDO_CSV" data.pseudo_loss_weight=1.0 output.checkpoint_dir="$QWEN_R7_CKPT_DIR" output.dir="$QWEN_R7_OUT_DIR"
) > "logs_checkpoint/r7_qwen_${JOB_TAG}.log" 2>&1 &
QWEN_PID=$!
ARM_PIDS+=("$QWEN_PID")

# --- Launch TrOCR -----------------------------------------------------------------
echo "Launching TrOCR-large-handwritten full fine-tune on allocated GPU $TROCR_GPU..."
(
  export HIP_VISIBLE_DEVICES=$TROCR_GPU
  exec python -m src.train_trocr --config configs/trocr_large.yaml \
      --resume --init-from "$TROCR_R5_GOLD" \
      --set data.pseudo_csv="$PSEUDO_CSV" data.pseudo_loss_weight=1.0 output.checkpoint_dir="$TROCR_R7_CKPT_DIR" output.dir="$TROCR_R7_OUT_DIR" model.attn_implementation=eager
) > "logs_checkpoint/r7_trocr_${JOB_TAG}.log" 2>&1 &
TROCR_PID=$!
ARM_PIDS+=("$TROCR_PID")

echo "Qwen PID=$QWEN_PID   TrOCR PID=$TROCR_PID"
echo "Tail logs with: tail -f logs_checkpoint/r7_qwen_${JOB_TAG}.log logs_checkpoint/r7_trocr_${JOB_TAG}.log"

# --- Wait for BOTH, even if one fails (`|| ...` keeps set -e from exiting) ----
QWEN_STATUS=0;  wait "$QWEN_PID"  || QWEN_STATUS=$?
echo "Qwen  exit code: $QWEN_STATUS  ($(date -Is))"
TROCR_STATUS=0; wait "$TROCR_PID" || TROCR_STATUS=$?
echo "TrOCR exit code: $TROCR_STATUS  ($(date -Is))"

if [ "$QWEN_STATUS" -ne 0 ] || [ "$TROCR_STATUS" -ne 0 ]; then
    echo "FATAL: One or both R7 training arms failed (Qwen=$QWEN_STATUS, TrOCR=$TROCR_STATUS)." >&2
    exit 1
fi

echo "== R7 training complete for both arms at $(date) =="
exit 0
