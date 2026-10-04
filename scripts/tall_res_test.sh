#!/bin/bash
# inference-only: fold-0 Qwen3-VL-8B (R8) on the tall (>200px) val lines at several line heights; run INSIDE job 1692 (overlap) on GPU 2
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for H in 256 384 512; do
  srun --jobid=1692 --overlap --ntasks=1 --cpus-per-task=8 --mem=48G bash -c "
    source slurm/common.sh >/dev/null; export HIP_VISIBLE_DEVICES=2 PYTHONPATH=\$PWD/.pylib_qwen3_checkpoint:\$PWD:\$PYTHONPATH
    python -m src.infer --config configs/qwen3vl_8b.yaml --checkpoint runs_checkpoint/BEST_qwen3vl8b_r8_f55_08997 --arm vlm \
      --manifest preds_checkpoint/val_tall_ids.csv --nbest 1 --set generate.num_beams=1 model.image_height=$H model.max_pixels=6000000 model.attn_implementation=sdpa \
      --out preds_checkpoint/tall_h${H}_nbest.csv --top1_out preds_checkpoint/tall_h${H}_top1.csv" > logs_checkpoint/tall_h${H}.log 2>&1
  echo "h$H done rc=$?" >> logs_checkpoint/tall_res.out
done
