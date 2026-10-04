#!/bin/bash
# R16 tall-crop experiments as overlap steps inside job 1692 (GPUs freed when r15_both and r15_long_all were stopped).
#   E1 (GPU 2): continue the R8 fold-0 LoRA on tall crops only        configs/qwen3vl_8b_r16_tallft.yaml    (--init-from BEST_qwen3vl8b_r8_f55_08997)
#   E2 (GPU 1): plain R8 recipe with tall crops sampled 3x             configs/qwen3vl_8b_r16_tallboost.yaml
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB=${1:-1692}
run() {   # run GPU NAME CONFIG [extra train args]
    local gpu=$1 name=$2 cfg=$3; shift 3
    srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=16 --mem=64G bash -c "
        source slurm/common.sh > /dev/null
        export HIP_VISIBLE_DEVICES=$gpu PYTHONPATH=\$PWD/.pylib_qwen3_checkpoint:\$PYTHONPATH
        for attempt in 1 2 3 4 5; do
            python -m src.train_vlm --config $cfg $* --resume --set && { echo 'R16 $name TRAIN FINISHED'; exit 0; }
            echo \"== $name attempt \$attempt failed at \$(date -Is); retrying in 120s ==\"; sleep 120
        done" >> logs_checkpoint/r16_${name}_${JOB}.log 2>&1
}
run 2 tallft configs/qwen3vl_8b_r16_tallft.yaml --init-from runs_checkpoint/BEST_qwen3vl8b_r8_f55_08997 &
run 1 tallboost configs/qwen3vl_8b_r16_tallboost.yaml &
wait
