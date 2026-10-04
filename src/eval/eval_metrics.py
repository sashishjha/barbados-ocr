"""
Local evaluation harness.

Contract with the organizers' scorer
------------------------------------
`Starters/*/eval_metrics.py` (identical in all three kits) computes:

    final_score = cer_weight * cer["score"] + wer_weight * wer["score"]
    # defaults in the shipped code: cer_weight=0.7, wer_weight=0.3

The competition *page* instead describes a 0.5 / 0.5 split. We trust the code,
so `final_07_03` is our PRIMARY model-selection metric, but every run also logs
`final_05_05` so that a single flag flip re-ranks experiments if the organizers
correct the page or the code before the deadline.

Everything is also logged under all three normalization regimes and both
weighting schemes, because the `compute_weighted_*` implementations were not
shipped (see src/eval/cer.py docstring).

Usage
-----
    python -m src.eval.eval_metrics --gt data/oof_ref.csv --pred submission.csv
    python -m src.eval.eval_metrics --config configs/base.yaml --gt ... --pred ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .cer import compute_weighted_cer
    from .wer import compute_weighted_wer
except ImportError:  # pragma: no cover
    from cer import compute_weighted_cer
    from wer import compute_weighted_wer


CER_WEIGHT_CODE = 0.7   # from organizers' eval_metrics.py
WER_WEIGHT_CODE = 0.3
CER_WEIGHT_PAGE = 0.5   # from competition page text
WER_WEIGHT_PAGE = 0.5


def evaluate_set(
    gt_csv_path: str,
    pred_csv_path: str,
    set_name: str = "SET",
    weight_factor: float = 0.5,
    normalization: str = "raw",
    scheme: str = "length_pow",
    verbose: bool = True,
) -> Dict:
    wer_result = compute_weighted_wer(
        gt_csv_path, pred_csv_path,
        weight_factor=weight_factor, normalization=normalization, scheme=scheme,
    )
    cer_result = compute_weighted_cer(
        gt_csv_path, pred_csv_path,
        weight_factor=weight_factor, normalization=normalization, scheme=scheme,
    )

    if wer_result["matched"] == 0:
        if verbose:
            print(f"===== {set_name} =====\nNo matching samples found!")
        return {
            "set": set_name, "cer": 0.0, "wer": 0.0,
            "cer_error": 1.0, "wer_error": 1.0,
            "final_07_03": 0.0, "final_05_05": 0.0,
            "matched": 0, "total_reference": wer_result["total_reference"],
        }

    cer_score = cer_result["score"]
    wer_score = wer_result["score"]
    final_code = CER_WEIGHT_CODE * cer_score + WER_WEIGHT_CODE * wer_score
    final_page = CER_WEIGHT_PAGE * cer_score + WER_WEIGHT_PAGE * wer_score

    out = {
        "set": set_name,
        "normalization": normalization,
        "scheme": scheme,
        "weight_factor": weight_factor,
        "cer": cer_score,
        "wer": wer_score,
        "cer_error": cer_result["error"],
        "wer_error": wer_result["error"],
        "final_07_03": final_code,
        "final_05_05": final_page,
        "matched": wer_result["matched"],
        "total_reference": wer_result["total_reference"],
    }

    if verbose:
        print(f"\n===== {set_name} [norm={normalization} scheme={scheme} wf={weight_factor}] =====")
        print(f"Samples: {out['matched']}/{out['total_reference']}")
        print(f"CER score: {cer_score:.4f}   (weighted CER error: {cer_result['error']:.4f})")
        print(f"WER score: {wer_score:.4f}   (weighted WER error: {wer_result['error']:.4f})")
        print(f"FINAL (0.7*CER + 0.3*WER)  <-- PRIMARY : {final_code:.4f}")
        print(f"FINAL (0.5*CER + 0.5*WER)  (page text) : {final_page:.4f}\n")
    return out


def length_bucket_report(
    gt_csv_path: str, pred_csv_path: str, weight_factor: float = 0.5,
    normalization: str = "raw", bins=(0, 45, 60, 75, 10_000),
) -> pd.DataFrame:
    """Per-length-bucket unweighted CER/WER. Long lines dominate the metric, so
    this is where regressions must be caught."""
    cer = compute_weighted_cer(gt_csv_path, pred_csv_path, weight_factor, normalization)
    wer = compute_weighted_wer(gt_csv_path, pred_csv_path, weight_factor, normalization)
    if cer["matched"] == 0:
        return pd.DataFrame()
    df = cer["per_sample"].merge(
        wer["per_sample"][["ID", "wer", "ref_words"]], on="ID", how="left"
    )
    df["bucket"] = pd.cut(df["ref_len"], bins=list(bins), right=False)
    rep = df.groupby("bucket", observed=True).agg(
        n=("ID", "count"),
        mean_ref_len=("ref_len", "mean"),
        cer=("cer", "mean"),
        wer=("wer", "mean"),
        weight_share=("weight", "sum"),
    )
    rep["weight_share"] = rep["weight_share"] / rep["weight_share"].sum()
    return rep.reset_index()


def full_report(gt_csv_path: str, pred_csv_path: str, weight_factor: float = 0.5) -> List[Dict]:
    """Every normalization x scheme combination. Use for run logging."""
    results = []
    for norm in ("raw", "strip_ws", "aggressive"):
        for scheme in ("length_pow", "micro", "uniform"):
            results.append(
                evaluate_set(
                    gt_csv_path, pred_csv_path,
                    set_name=f"{norm}/{scheme}",
                    weight_factor=weight_factor,
                    normalization=norm, scheme=scheme, verbose=False,
                )
            )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Reference CSV with ID,Target")
    ap.add_argument("--pred", required=True, help="Prediction CSV with ID,Target")
    ap.add_argument("--set_name", default="OOF")
    ap.add_argument("--weight_factor", type=float, default=0.5)
    ap.add_argument("--normalization", default="raw",
                    choices=["raw", "strip_ws", "aggressive"])
    ap.add_argument("--scheme", default="length_pow",
                    choices=["length_pow", "micro", "uniform"])
    ap.add_argument("--buckets", action="store_true", help="Print length-bucket breakdown")
    ap.add_argument("--full", action="store_true", help="Sweep all norm x scheme combos")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    for p in (args.gt, args.pred):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    primary = evaluate_set(
        args.gt, args.pred, args.set_name, args.weight_factor,
        args.normalization, args.scheme,
    )

    payload = {"primary": primary}

    if args.buckets:
        rep = length_bucket_report(args.gt, args.pred, args.weight_factor, args.normalization)
        print("Per-length-bucket breakdown (unweighted CER/WER, plus each bucket's")
        print("share of the total metric weight):")
        print(rep.to_string(index=False))
        payload["buckets"] = rep.to_dict(orient="records")

    if args.full:
        payload["sweep"] = full_report(args.gt, args.pred, args.weight_factor)
        print("\nNormalization x scheme sweep (final_07_03):")
        for r in payload["sweep"]:
            print(f"  {r['set']:<24} {r['final_07_03']:.4f}   (0.5/0.5: {r['final_05_05']:.4f})")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"[DONE] wrote {args.json_out}")


if __name__ == "__main__":
    main()
