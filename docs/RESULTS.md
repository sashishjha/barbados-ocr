# Results and findings

Competition: Zindi *R.O.A.D. Barbados Historic Handwriting Challenge* — transcribe single text-line crops from 17th–18th-century
Barbados records. Metric: `1 − (0.5·CER + 0.5·WER)`, case-sensitive; the leaderboard's "weighted" columns are mean edit counts per line.
4,054 labelled lines (5 folds; fold 0 = 811 lines used for validation), 1,374 test lines. Final public score **0.92417**
(the leader was at 0.9485).

## What the final submission is
Character-level ROVER vote (with a pre-filter for degenerate candidates and a runaway-line guard) over 6 members:

| Member | Model |
|---|---|
| Qwen3-VL-32B, LoRA r32 | trained on train folds |
| Qwen3-VL-8B, vision-tower LoRA r128 | trained on all labelled data (seed 1) |
| Qwen3-VL-8B, vision-tower LoRA r128 | all labelled data (seed 7) |
| Qwen2.5-VL-7B, vision-tower LoRA r128 | all labelled data |
| Qwen2.5-VL-7B, LoRA r32 | trained on train folds |
| TrOCR-large-handwritten, fine-tuned | weight 2 |

All Qwen members were trained on real labels only. The TrOCR member comes from an earlier self-training round that used
pseudo-labelled test lines (see the pseudo-label finding below).

## Leaderboard progression (public)
| Step | Public score |
|---|---|
| ROVER of Qwen2.5-VL-7B + TrOCR (first VLM ensemble) | 0.898 |
| 4-model ROVER of round-8 models (Qwen3-VL-8B, Qwen2.5-VL-7B ×2, TrOCR) | 0.9154 |
| + Qwen3-VL-32B (after repairing one runaway output line) | 0.9187 |
| 6 members, all-data twins | 0.9221 |
| TrOCR weight 2 | 0.92267 |
| vision-LoRA r128 members, equal weights | 0.92300 |
| vision-LoRA r128 members, TrOCR weight 2 | **0.92417** |

## Single models (fold-0 validation, 0.5·CER+0.5·WER)
| Model | Validation |
|---|---|
| TrOCR-large-handwritten | 0.851–0.878 |
| InternVL3.5-8B | 0.8395 |
| Qwen2.5-VL-32B | 0.8933 |
| Qwen2.5-VL-7B, LoRA r32 | 0.8986 |
| Qwen3-VL-8B, LoRA r32 | 0.8987–0.9034 |
| Qwen3-VL-32B, LoRA r32 | 0.9002 |
| Qwen2.5-VL-7B, vision-tower LoRA r128 | 0.9030 (epoch 2 of 3) |
| **Qwen3-VL-8B, vision-tower LoRA r128** | **0.9054** |

The only recipe change that clearly improved single models was **giving the vision tower LoRA capacity (rank 128)**; it
transferred from Qwen3-VL to Qwen2.5-VL. A 4× bigger model (32B) gave no gain over 8B.

## Things that did not help (all measured)
- More seeds/members of an existing family in the vote; family-balanced or tuned weights; word-level ROVER; MBR; beam/height TTA.
- Higher input resolution (384 px); test-set pseudo-labels (validation up, leaderboard down — the model echoes its labeller).
- Selecting among n-best candidates with teacher-forced log-probs from several models, multi-view likelihoods or a learned ranker
  (the candidate pool's oracle is 0.953 vs 0.916 for the vote, but no selector found it).
- Full fine-tuning of the vision tower: layer-decayed LR 3e-5 (epoch 1: 0.8904) and uniform LR 5e-5 with heavy augmentation
  (epoch 1: 0.8601) both lost to vision LoRA r128 (0.8967 at epoch 1).
- Label-convention handling, spacing rules, ink-band cropping, abbreviation expansion.
- Document context: image-nearest training lines contain a line's rare words only 1.6% of the time (random 0.5%); recoverable
  fixes were ≈ +0.001 at best (`scripts/doc_context_probe.py`).

## Metric and engineering findings
- The official metric does **not** cap per-line error: one degenerate repeated-token line moved the leaderboard 0.9005 → 0.9187 once
  replaced. Every submission builder here applies `prefilter_candidates` + `guard_runaway`.
- Validation differences below ≈ 0.005 did not predict public-leaderboard direction (811 validation lines, ~30% public split).
- All-data twins (train + validation fold) gained ≈ +0.002–0.003 on the leaderboard.
