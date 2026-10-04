#!/usr/bin/env python3
"""
Day-1 Diagnostic Suite (Expert Roadmap §1)
Runs 5 diagnostics on val set using existing OOF / test predictions:
  1. Oracle CER  -- best of 3 arms per line
  2. Train-val vs LB gap (prints a warning if suspicious)
  3. Segmentation audit -- contact sheet of 100 random test crops
  4. CER vs line-length scatter
  5. Error taxonomy -- Levenshtein confusion matrix

Outputs:
  report/diagnostics/oracle_cer.txt
  report/diagnostics/length_scatter.csv
  report/diagnostics/confusion_matrix.csv
  report/diagnostics/contact_sheet.png    (100 random test crops)
  report/diagnostics/error_taxonomy.txt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import Levenshtein  # pip install python-Levenshtein


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, ref) / len(ref)


def wer(pred: str, ref: str) -> float:
    p, r = pred.split(), ref.split()
    if not r:
        return 0.0 if not p else 1.0
    return Levenshtein.distance(" ".join(p), " ".join(r)) / len(r)


def blended(pred: str, ref: str) -> float:
    c = 1.0 - cer(pred, ref)
    w = 1.0 - wer(pred, ref)
    return 0.7 * c + 0.3 * w


def load_preds(path: str) -> dict[str, str]:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return dict(zip(df["ID"].astype(str), df["Target"].fillna("").astype(str)))


# ---------------------------------------------------------------------------
# Diagnostic 1 — Oracle CER
# ---------------------------------------------------------------------------

def diag_oracle(gt: dict, arm_preds: list[dict], names: list[str], out_dir: Path):
    print("\n=== Diagnostic 1: Oracle CER ===")
    rows = []
    arm_cers = {n: [] for n in names}
    oracle_cers = []
    # NOTE: We do NOT load ensemble_test_pp.csv here. That file has TEST set IDs
    # and cannot be compared against val GT. The ensemble column was bogus (always 100% error).
    # The real question is: Is the gap between oracle and best single arm large?

    for id_, ref in gt.items():
        arm_scores = []
        for name, preds in zip(names, arm_preds):
            pred = preds.get(id_, "")
            c = cer(pred, ref)
            arm_cers[name].append(c)
            arm_scores.append((c, pred))
        oracle_c = min(s for s, _ in arm_scores)
        oracle_cers.append(oracle_c)
        rows.append({"id": id_, "oracle_cer": oracle_c,
                     **{f"{n}_cer": c for n, (c, _) in zip(names, arm_scores)}})

    best_arm = min(names, key=lambda n: np.mean(arm_cers[n]))
    best_arm_cer = np.mean(arm_cers[best_arm])

    report = []
    report.append(f"Lines evaluated: {len(oracle_cers)}")
    report.append(f"{'Arm':<25} {'Mean CER':>10}")
    report.append("-" * 38)
    for n in names:
        report.append(f"{n:<25} {np.mean(arm_cers[n]):>10.4f}")
    report.append(f"{'Oracle (best per line)':<25} {np.mean(oracle_cers):>10.4f}")

    gap = best_arm_cer - np.mean(oracle_cers)
    report.append(f"\nBest single arm ({best_arm}) - Oracle gap: {gap:.4f}")
    if gap > 0.03:
        report.append(">>> FUSION IS THE PROBLEM: a good ensemble could gain >3% CER.")
    else:
        report.append(">>> Arms are individually strong. Oracle ≈ best arm; fusion gains are limited.")
        report.append("    Root cause is likely domain shift (test has different scribes).")

    text = "\n".join(report)
    print(text)
    (out_dir / "oracle_cer.txt").write_text(text)

    pd.DataFrame(rows).to_csv(out_dir / "oracle_cer_per_line.csv", index=False)
    print(f"Saved to {out_dir}/oracle_cer.txt")


# ---------------------------------------------------------------------------
# Diagnostic 2 — Train/val vs LB gap warning
# ---------------------------------------------------------------------------

def diag_lb_gap(val_cer: float, lb_score: float, out_dir: Path):
    print("\n=== Diagnostic 2: Train-val vs LB gap ===")
    lb_cer_approx = 1.0 - (lb_score - 0.3 * (1.0 - lb_score * 0.4)) / 0.7
    gap = lb_cer_approx - val_cer
    lines = [
        f"Val CER (held-out):  {val_cer:.4f}",
        f"LB score:            {lb_score:.4f}  (approx LB CER: {lb_cer_approx:.4f})",
        f"Gap (LB - val):      {gap:.4f}",
    ]
    if gap > 0.05:
        lines.append(">>> WARNING: Large gap suggests segmentation or preprocessing mismatch.")
        lines.append("    Action: Run Diagnostic 3 (segmentation audit) immediately.")
    else:
        lines.append("Val and LB are close — generalization looks good.")
    text = "\n".join(lines)
    print(text)
    (out_dir / "lb_gap.txt").write_text(text)


# ---------------------------------------------------------------------------
# Diagnostic 3 — Segmentation contact sheet
# ---------------------------------------------------------------------------

def diag_contact_sheet(image_dir: str, test_csv: str, out_dir: Path, n=100, seed=42):
    print("\n=== Diagnostic 3: Segmentation contact sheet ===")
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not installed, skipping contact sheet. Run: pip install Pillow")
        return

    df = pd.read_csv(test_csv)
    sample = df.sample(min(n, len(df)), random_state=seed)
    img_dir = Path(image_dir)

    cols, rows_grid = 5, 20
    thumb_w, thumb_h = 300, 80
    canvas = Image.new("RGB", (cols * thumb_w, rows_grid * thumb_h), (200, 200, 200))

    found = 0
    for i, (_, row) in enumerate(sample.iterrows()):
        img_id = str(row["ID"])
        # try common extensions
        for ext in [".jpg", ".jpeg", ".png"]:
            img_path = img_dir / f"{img_id}{ext}"
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.thumbnail((thumb_w, thumb_h))
                    x = (found % cols) * thumb_w
                    y = (found // cols) * thumb_h
                    canvas.paste(img, (x, y))
                    found += 1
                except Exception:
                    pass
                break

    out_path = out_dir / "contact_sheet.png"
    canvas.save(out_path)
    print(f"Contact sheet saved: {out_path}  ({found}/{n} images found)")
    if found == 0:
        print("  No images found — check --image_dir path.")


# ---------------------------------------------------------------------------
# Diagnostic 4 — CER vs line length scatter
# ---------------------------------------------------------------------------

def diag_length_scatter(gt: dict, arm_preds: list[dict], names: list[str], out_dir: Path):
    print("\n=== Diagnostic 4: CER vs line length ===")
    rows = []
    for id_, ref in gt.items():
        for name, preds in zip(names, arm_preds):
            pred = preds.get(id_, "")
            rows.append({
                "id": id_, "arm": name,
                "ref_len": len(ref),
                "pred_len": len(pred),
                "cer": cer(pred, ref),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "length_scatter.csv", index=False)

    # Print bucketed summary
    df["len_bucket"] = pd.cut(df["ref_len"], bins=[0, 20, 40, 60, 80, 100, 9999],
                              labels=["0-20", "21-40", "41-60", "61-80", "81-100", "100+"])
    pivot = df.groupby(["arm", "len_bucket"])["cer"].mean().unstack("len_bucket")
    print(pivot.to_string())

    # Alert on VLM looping
    vlm_long = df[(df["arm"].str.contains("vlm")) & (df["ref_len"] > 60)]["cer"].mean()
    if vlm_long > 0.25:
        print(f"\n>>> WARNING: VLM CER at len>60 is {vlm_long:.3f} — possible decoding loop/truncation!")

    print(f"Saved to {out_dir}/length_scatter.csv")


# ---------------------------------------------------------------------------
# Diagnostic 5 — Error taxonomy / confusion matrix
# ---------------------------------------------------------------------------

def diag_error_taxonomy(gt: dict, arm_preds: list[dict], names: list[str], out_dir: Path):
    print("\n=== Diagnostic 5: Error taxonomy ===")
    # Per arm: count substitutions, insertions, deletions, and top char confusions
    for name, preds in zip(names, arm_preds):
        sub_counter: Counter = Counter()
        ins_counter: Counter = Counter()
        del_counter: Counter = Counter()
        total_sub = total_ins = total_del = 0

        for id_, ref in gt.items():
            pred = preds.get(id_, "")
            ops = Levenshtein.editops(pred, ref)
            for op, i, j in ops:
                if op == "replace":
                    sub_counter[(pred[i] if i < len(pred) else "?",
                                 ref[j] if j < len(ref) else "?")] += 1
                    total_sub += 1
                elif op == "insert":
                    ins_counter[ref[j] if j < len(ref) else "?"] += 1
                    total_ins += 1
                elif op == "delete":
                    del_counter[pred[i] if i < len(pred) else "?"] += 1
                    total_del += 1

        lines = [
            f"\n--- Arm: {name} ---",
            f"Substitutions: {total_sub}  Insertions: {total_ins}  Deletions: {total_del}",
            f"Top 20 substitutions (pred→ref):",
        ]
        for (p, r), cnt in sub_counter.most_common(20):
            lines.append(f"  '{p}' → '{r}' : {cnt}")
        lines.append(f"Top 10 deleted chars: {del_counter.most_common(10)}")
        lines.append(f"Top 10 inserted chars: {ins_counter.most_common(10)}")
        print("\n".join(lines))

    # Also save a combined confusion matrix CSV
    all_subs = Counter()
    for name, preds in zip(names, arm_preds):
        for id_, ref in gt.items():
            pred = preds.get(id_, "")
            ops = Levenshtein.editops(pred, ref)
            for op, i, j in ops:
                if op == "replace":
                    all_subs[(pred[i] if i < len(pred) else "?",
                               ref[j] if j < len(ref) else "?")] += 1

    conf_rows = [{"pred_char": p, "ref_char": r, "count": c}
                 for (p, r), c in all_subs.most_common(200)]
    pd.DataFrame(conf_rows).to_csv(out_dir / "confusion_matrix.csv", index=False)
    print(f"\nSaved confusion matrix to {out_dir}/confusion_matrix.csv")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Day-1 Diagnostic Suite")
    ap.add_argument("--gt_csv", default="data/oof_ref.csv",
                    help="Ground-truth CSV (ID,Target) for the val split")
    ap.add_argument("--trocr_pred", default="preds_checkpoint/trocr_test.csv")
    ap.add_argument("--vlm_pred", default="preds_checkpoint/vlm_test.csv")
    ap.add_argument("--ctc_pred", default="preds_checkpoint/ctc_test.csv")
    ap.add_argument("--image_dir", default="data/images",
                    help="Path to image files for contact sheet")
    ap.add_argument("--test_csv", default="data/Test.csv")
    ap.add_argument("--lb_score", type=float, default=0.788,
                    help="Current public LB score for gap analysis")
    ap.add_argument("--out_dir", default="report/diagnostics")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GT and predictions
    gt = load_preds(args.gt_csv)
    arm_preds = [
        load_preds(args.trocr_pred),
        load_preds(args.vlm_pred),
        load_preds(args.ctc_pred),
    ]
    names = ["trocr", "vlm", "ctc"]

    # Filter to IDs present in GT
    gt = {k: v for k, v in gt.items() if k in arm_preds[0] or k in arm_preds[1] or k in arm_preds[2]}
    if not gt:
        print("WARNING: No IDs matched between GT and predictions.")
        print("  Check that --gt_csv has the same IDs as the prediction CSVs.")
        print("  If using test predictions for diagnostics, use --gt_csv data/oof_ref.csv")
        sys.exit(1)

    print(f"Running diagnostics on {len(gt)} lines.")

    # Oracle CER — needs GT
    val_cers = []
    for id_, ref in gt.items():
        best = min(cer(arm_preds[i].get(id_, ""), ref) for i in range(3))
        val_cers.append(best)
    diag_oracle(gt, arm_preds, names, out_dir)
    diag_lb_gap(float(np.mean(val_cers)), args.lb_score, out_dir)
    diag_contact_sheet(args.image_dir, args.test_csv, out_dir)
    diag_length_scatter(gt, arm_preds, names, out_dir)
    diag_error_taxonomy(gt, arm_preds, names, out_dir)

    print(f"\n=== All diagnostics done. Reports in {out_dir}/ ===")


if __name__ == "__main__":
    main()
