#!/bin/bash
# Unattended orchestration inside job 1635 (R13): keep all 3 GPUs useful until R13 ends (~05:00 3 Oct) and
# do the InternVL3.5-8B work as overlap steps, so job 37_train_r14 only has to mop up.
#   GPU 0: InternVL fold-0 head start (already running) -> greedy val + test inference when training is done
#   GPU 1: after the Qwen2.5-7B seed-17 arm finishes -> InternVL ALL-data head start -> test inference
# Greedy decoding (HF beam search scrambles InternVL tiles). Run: nohup setsid bash slurm/38_internvl_overnight.sh &
set -u
JOB=1635
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=logs_checkpoint/internvl_overnight_${JOB}.log
log() { echo "$(date -Is) $*" >> "$LOG"; }
alive() { squeue -h -j $JOB -o %T | grep -q RUNNING; }
wait_for() {   # wait_for FILE PATTERN
    until grep -q "$2" "$1" 2>/dev/null; do alive || { log "job $JOB ended while waiting for '$2' in $1"; return 1; }; sleep 60; done
}
infer() {      # infer NAME GPU CONFIG "SPLITS"
    local name=$1 gpu=$2 cfg=$3 splits=$4 ckpt
    ckpt=$(python3 -c "import yaml,sys; print('x')" >/dev/null 2>&1; .venv/bin/python -c "from src.utils.common import load_config as l; print(l('$cfg')['output']['checkpoint_dir'])" 2>/dev/null | tail -1)
    log "inference $name ($splits) on GPU $gpu from $ckpt/best_checkpoint"
    srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=16 --mem=64G bash -c '
        source slurm/common.sh > /dev/null
        export HIP_VISIBLE_DEVICES='"$gpu"' PYTHONPATH="$PWD/.pylib_qwen3_checkpoint:$PYTHONPATH"
        for split in '"$splits"'; do
            man=$([ $split = val ] && echo data/derived_checkpoint/val_fold0_ids.csv || echo data/Test.csv)
            python -m src.infer --config '"$cfg"' --checkpoint '"$ckpt"'/best_checkpoint --arm vlm --manifest "$man" \
                --nbest 1 --set generate.num_beams=1 \
                --out "preds_checkpoint/'"$name"'_${split}_nbest_'"$JOB"'.csv" \
                --top1_out "preds_checkpoint/'"$name"'_${split}_top1_'"$JOB"'.csv" || exit 1
        done
        echo "INFER '"$name"' DONE"
    ' >> "logs_checkpoint/overnight_infer_${name}_${JOB}.log" 2>&1
    log "inference $name finished rc=$?"
}

( # branch 1: fold-0 InternVL
  wait_for logs_checkpoint/pre_internvl35_8b_r14_${JOB}.log "PRETRAIN internvl35_8b_r14 FINISHED" && \
  infer internvl35_8b_r14 0 configs/internvl35_8b_r14.yaml "val test"
) &
( # branch 2: all-data InternVL on GPU 1 once the 7B seed-17 arm has released it
  wait_for logs_checkpoint/r13_qwen7b_r13all_s17_${JOB}.log "ARM qwen7b_r13all_s17 DONE" || exit 0
  log "starting all-data InternVL head start on GPU 1"
  bash slurm/36_overlap_pretrain.sh $JOB 1 internvl35_8b_r14_all configs/internvl35_8b_r14_all.yaml .pylib_qwen3_checkpoint
  grep -q "PRETRAIN internvl35_8b_r14_all FINISHED" logs_checkpoint/pre_internvl35_8b_r14_all_${JOB}.log && \
  infer internvl35_8b_r14_all 1 configs/internvl35_8b_r14_all.yaml "test"
) &
wait
log "overnight orchestration finished"
