"""
Final submission assembly with hard guarantees.

The competition scores a missing, empty, or invalid prediction as fully wrong,
so this script refuses to write a file unless:

  * every ID in Test.csv is present exactly once,
  * no Target is empty or whitespace-only,
  * no Target contains a newline or carriage return (would break CSV parsing),
  * the column order is exactly ID,Target.

Fallback chain for any missing/empty row (all automated):
  primary prediction -> fallback CSV(s) in order -> most frequent training
  line-opening string. The last resort is derived from TRAIN statistics, so it
  is still "provided dataset only" and is never a hand-written guess.

Also appends a row to submissions/log.csv recording the file hash, config hash,
OOF score and any notes, which is what the 48h code review will want to see.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from datetime import datetime, timezone
from typing import List

import pandas as pd

from src.utils.common import get_logger

log = get_logger("submit")

LOG_COLUMNS = [
    "timestamp", "submission_file", "file_sha1", "config_hash", "method",
    "oof_final_07_03", "oof_final_05_05", "lb_public", "notes",
]


def most_common_train_line(train_csv: str) -> str:
    df = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
    vc = df["Target"].fillna("").str.strip()
    vc = vc[vc != ""]
    return vc.value_counts().idxmax() if len(vc) else "the"


def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def build(
    test_csv: str,
    pred_csv: str,
    out_csv: str,
    fallbacks: List[str] | None = None,
    train_csv: str | None = None,
) -> pd.DataFrame:
    test = pd.read_csv(test_csv, dtype=str, keep_default_na=False)
    if "ID" not in test.columns:
        test = test.rename(columns={test.columns[0]: "ID"})
    test["ID"] = test["ID"].str.strip()
    test = test[test["ID"] != ""][["ID"]].drop_duplicates()

    pred = pd.read_csv(pred_csv, dtype=str, keep_default_na=False)
    pred = pred[["ID", "Target"]].drop_duplicates(subset="ID", keep="last")

    sub = test.merge(pred, on="ID", how="left")
    sub["Target"] = sub["Target"].fillna("")

    n_missing_primary = int((sub["Target"].str.strip() == "").sum())

    for fb_path in fallbacks or []:
        need = sub["Target"].str.strip() == ""
        if not need.any():
            break
        if not os.path.exists(fb_path):
            log.warning("fallback missing: %s", fb_path)
            continue
        fb = pd.read_csv(fb_path, dtype=str, keep_default_na=False)
        fb = fb[["ID", "Target"]].drop_duplicates(subset="ID", keep="last")
        m = sub.loc[need, ["ID"]].merge(fb, on="ID", how="left")
        filled = m["Target"].fillna("").values
        sub.loc[need, "Target"] = filled
        sub["Target"] = sub["Target"].fillna("")
        log.info("fallback %s filled %d rows",
                 fb_path, int((pd.Series(filled).str.strip() != "").sum()))

    still = sub["Target"].str.strip() == ""
    if still.any():
        last = most_common_train_line(train_csv) if train_csv else "the"
        log.warning("%d rows still empty; filling with the most frequent TRAIN "
                    "line: %r", int(still.sum()), last)
        sub.loc[still, "Target"] = last

    # Sanitize: strip newlines/tabs, collapse whitespace.
    sub["Target"] = (
        sub["Target"].astype(str)
        .str.replace(r"[\r\n\t]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # ---- hard assertions
    assert list(sub.columns) == ["ID", "Target"], f"bad columns {list(sub.columns)}"
    assert len(sub) == len(test), f"row count {len(sub)} != test {len(test)}"
    assert sub["ID"].is_unique, "duplicate IDs in submission"
    assert set(sub["ID"]) == set(test["ID"]), "ID set mismatch vs Test.csv"
    empty = sub["Target"].str.strip() == ""
    assert not empty.any(), f"{int(empty.sum())} empty predictions remain"
    assert not sub["Target"].str.contains(r"[\r\n]").any(), "newline in a prediction"

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    sub.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    log.info("submission OK: %d rows -> %s", len(sub), out_csv)
    log.info("  empty from primary model : %d (filled by fallback chain)", n_missing_primary)
    log.info("  mean prediction length   : %.1f chars", sub["Target"].str.len().mean())
    log.info("  sha1                     : %s", sha1_file(out_csv))
    return sub


def append_log(log_csv: str, **fields) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(log_csv)), exist_ok=True)
    row = {c: fields.get(c, "") for c in LOG_COLUMNS}
    row["timestamp"] = fields.get("timestamp") or datetime.now(timezone.utc).isoformat()
    exists = os.path.exists(log_csv)
    with open(log_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow(row)
    log.info("logged submission -> %s", log_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default="data/Test.csv")
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--fallbacks", nargs="*", default=[])
    ap.add_argument("--train_csv", default="data/derived_checkpoint/train_clean.csv")
    ap.add_argument("--log_csv", default="submissions/log.csv")
    ap.add_argument("--config_hash", default="")
    ap.add_argument("--method", default="")
    ap.add_argument("--oof_final_07_03", default="")
    ap.add_argument("--oof_final_05_05", default="")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    build(a.test_csv, a.pred_csv, a.out_csv, a.fallbacks, a.train_csv)
    append_log(a.log_csv, submission_file=a.out_csv, file_sha1=sha1_file(a.out_csv),
               config_hash=a.config_hash, method=a.method,
               oof_final_07_03=a.oof_final_07_03, oof_final_05_05=a.oof_final_05_05,
               notes=a.notes)


if __name__ == "__main__":
    main()
