#!/bin/bash
#SBATCH --job-name=bbd_pseudo
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G

#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/pseudo_%j.out
#SBATCH --error=logs_checkpoint/pseudo_%j.err
#
# ONE round of the officially-sanctioned pseudo-labelling loop (17 Aug 2026).
#
#   a) infer TEST with two DIVERSE arms, beam search, n-best
#   b) filter by coded thresholds only (cross-model agreement, beam stability,
#      confidence percentile, length/charset sanity)
#   c) retrain with accepted pseudo-labels capped at 30% and loss-weighted 0.5
#   d) re-evaluate OOF. Only keep this round if the weighted metric IMPROVED.
#
# No human looks at a test image at any point. There is no manual-override path.

mkdir -p logs_checkpoint preds_checkpoint
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

ROUND=${ROUND:-1}
FOLD=${FOLD:-0}

echo "### (a) test inference, two arms in parallel on gpu0 / gpu1"
ROCR_VISIBLE_DEVICES=0 python -m src.infer \
    --config configs/trocr_large.yaml \
    --checkpoint runs_checkpoint/trocr_large_checkpoints/best_checkpoint \
    --manifest data/Test.csv --out preds_checkpoint/trocr_test_nbest_r${ROUND}.csv --nbest 5 &
P1=$!
ROCR_VISIBLE_DEVICES=1 python -m src.infer \
    --config configs/qwen25vl_lora.yaml \
    --checkpoint runs_checkpoint/qwen25vl_lora_checkpoints/best_checkpoint \
    --manifest data/Test.csv --out preds_checkpoint/vlm_test_nbest_r${ROUND}.csv --nbest 5 &
P2=$!
wait $P1 || { echo "trocr inference failed"; exit 1; }
wait $P2 || { echo "vlm inference failed"; exit 1; }

echo "### (b) coded-threshold filtering"
python -m src.pseudo_label \
    --candidates preds_checkpoint/trocr_test_nbest_r${ROUND}.csv preds_checkpoint/vlm_test_nbest_r${ROUND}.csv \
    --train_csv data/derived_checkpoint/train_clean.csv \
    --out_csv data/derived_checkpoint/pseudo_round${ROUND}.csv \
    --audit_csv report/pseudo_round${ROUND}_audit.csv \
    --max_beam_distance 0.10 --score_percentile 40

echo "### (c) retrain with pseudo-labels mixed in (capped, down-weighted)"
ROCR_VISIBLE_DEVICES=0 python -m src.train_vlm \
    --config configs/pseudo_round1.yaml --fold "${FOLD}" --resume \
    --set data.pseudo_csv=data/derived_checkpoint/pseudo_round${ROUND}.csv \
          output.dir=runs_checkpoint/qwen25vl_pseudo${ROUND} \
          output.checkpoint_dir=runs_checkpoint/qwen25vl_pseudo${ROUND}_checkpoints

echo "### (d) OOF comparison - decide whether to keep this round"
python -m src.eval.eval_metrics \
    --gt runs_checkpoint/qwen25vl_pseudo${ROUND}/val_ref_fold${FOLD}.csv \
    --pred runs_checkpoint/qwen25vl_pseudo${ROUND}/val_pred_fold${FOLD}.csv \
    --set_name "PSEUDO_R${ROUND}" --buckets --full \
    --json_out report/pseudo_round${ROUND}_oof.json

echo "COMPARE the final_07_03 above against the pre-pseudo baseline in"
echo "submissions/log.csv. Run another round ONLY if it improved."

report_disk
