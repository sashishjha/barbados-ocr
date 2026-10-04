"""
Optuna hyperparameter search (officially confirmed as allowed; this is NOT
AutoML - the search space and the model architecture are specified by us).

Budget-capped by design: the cluster allows 3 GPUs and 2 concurrent jobs, so an
Optuna worker rides along on a spare GPU inside an existing allocation rather
than occupying a job slot of its own (see slurm/12_parallel_arms.sbatch).

Two modes:
  --mode train    search LR / LoRA rank / augmentation strength by launching
                  short training runs (few epochs, one fold) and reading the
                  OOF weighted metric back from the run's val prediction CSV.
  --mode decode   search decoding hyperparameters (beam width, length penalty)
                  and ensemble/post-processing knobs against ALREADY-GENERATED
                  n-best candidate files. Very cheap - do this one first, it
                  usually pays for itself in minutes.

Storage is a SQLite file so a killed 24h job resumes the same study.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict

import pandas as pd

from src.utils.common import get_logger, load_config

log = get_logger("optuna")


# --------------------------------------------------------------------------
def _score_csv(ref_csv: str, pred_csv: str, cfg: Dict) -> float:
    from src.eval.eval_metrics import evaluate_set

    r = evaluate_set(ref_csv, pred_csv, "TRIAL",
                     weight_factor=cfg["eval"].get("weight_factor", 0.5),
                     normalization=cfg["eval"].get("normalization", "raw"),
                     scheme=cfg["eval"].get("scheme", "length_pow"),
                     verbose=False)
    return float(r[cfg["eval"].get("primary", "final_07_03")])


def objective_train(trial, args, cfg):
    """Short training run per trial. Keep epochs small; this is for ranking."""
    lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
    aug_strength = trial.suggest_float("aug_strength_start", 0.2, 1.0)
    overrides = [f"train.lr={lr}", f"train.aug_strength_start={aug_strength}",
                 f"train.epochs={args.trial_epochs}"]

    if cfg.get("arm") == "vlm" or cfg.get("model", {}).get("use_lora"):
        r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
        overrides += [f"model.lora_r={r}", f"model.lora_alpha={2*r}"]
    if cfg.get("stackmix", {}).get("enabled"):
        overrides.append(f"stackmix.p={trial.suggest_float('stackmix_p', 0.0, 0.6)}")

    out_dir = os.path.join(args.study_dir, f"trial_{trial.number:03d}")
    overrides += [f"output.dir={out_dir}",
                  f"output.checkpoint_dir={out_dir}_checkpoints"]

    entry = {"trocr": "src.train_trocr", "vlm": "src.train_vlm",
             "ctc": "src.train_ctc"}[cfg["arm"]]
    cmd = [sys.executable, "-m", entry, "--config", args.config,
           "--fold", str(args.fold), "--set", *overrides]
    log.info("trial %d: %s", trial.number, " ".join(cmd))
    subprocess.run(cmd, check=True)

    pred = os.path.join(out_dir, f"val_pred_fold{args.fold}.csv")
    ref = os.path.join(out_dir, f"val_ref_fold{args.fold}.csv")
    if not (os.path.exists(pred) and os.path.exists(ref)):
        raise RuntimeError(f"trial {trial.number} produced no validation output")
    return _score_csv(ref, pred, cfg)


def objective_decode(trial, args, cfg):
    """Search ensemble + post-processing knobs over precomputed n-best files."""
    from src.ensemble import combine
    from src.postprocess import (PostprocessConfig, TrainLexicon, apply_all,
                                 ConfusionRules)

    method = trial.suggest_categorical("method", ["rover", "mbr", "lexicon"])
    nbest = trial.suggest_int("nbest", 1, 5)
    weights = {}
    for a in args.arms:
        weights[a] = trial.suggest_float(f"w_{a}", 0.3, 2.0)
    lex_beta = trial.suggest_float("lex_beta", 0.0, 1.0)
    min_word_freq = trial.suggest_int("min_word_freq", 1, 6)
    use_confusion = trial.suggest_categorical("confusion", [True, False])

    lex = TrainLexicon.fit(args.train_csv)
    df = combine(args.candidates, method, weights, nbest, lex, 1.0, lex_beta)

    pcfg = PostprocessConfig(confusion=use_confusion, min_word_freq=min_word_freq)
    rules = ConfusionRules(lex, pcfg)
    df["Target"] = [apply_all(t, lex, rules, pcfg)[0] for t in df["Target"]]

    tmp = os.path.join(args.study_dir, f"decode_trial_{trial.number:03d}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(tmp)), exist_ok=True)
    df.to_csv(tmp, index=False)
    return _score_csv(args.oof_ref, tmp, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="decode", choices=["train", "decode"])
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--timeout_min", type=int, default=240, help="budget cap")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--trial_epochs", type=int, default=2)
    ap.add_argument("--study_dir", default="runs/optuna")
    ap.add_argument("--study_name", default=None)
    # decode mode
    ap.add_argument("--candidates", nargs="*", default=[])
    ap.add_argument("--arms", nargs="*", default=["trocr", "vlm", "ctc"])
    ap.add_argument("--oof_ref", default="data/derived_checkpoint/oof_ref.csv")
    ap.add_argument("--train_csv", default="data/derived_checkpoint/train_clean.csv")
    args = ap.parse_args()

    import optuna

    cfg = load_config(args.config)
    os.makedirs(args.study_dir, exist_ok=True)
    name = args.study_name or f"{cfg.get('arm','x')}_{args.mode}"
    storage = f"sqlite:///{os.path.join(args.study_dir, name)}.db"

    study = optuna.create_study(
        study_name=name, storage=storage, direction="maximize",
        load_if_exists=True,          # resumable after a 24h kill
        sampler=optuna.samplers.TPESampler(seed=cfg.get("seed", 42)),
    )

    fn = (lambda t: objective_train(t, args, cfg)) if args.mode == "train" \
        else (lambda t: objective_decode(t, args, cfg))
    study.optimize(fn, n_trials=args.n_trials, timeout=args.timeout_min * 60)

    log.info("best value (%s): %.5f", cfg["eval"].get("primary"), study.best_value)
    log.info("best params: %s", study.best_params)
    with open(os.path.join(args.study_dir, f"{name}_best.json"), "w") as f:
        json.dump({"value": study.best_value, "params": study.best_params}, f, indent=2)

    pd.DataFrame([{**t.params, "value": t.value, "state": str(t.state)}
                  for t in study.trials]).to_csv(
        os.path.join(args.study_dir, f"{name}_trials.csv"), index=False)


if __name__ == "__main__":
    main()
