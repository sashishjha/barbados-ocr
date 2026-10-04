"""
Unit tests that run WITHOUT the images and WITHOUT a GPU.

These exist so the eval harness, ensembling, post-processing and submission
guarantees can be verified on a login node before burning a job slot.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ensemble import combine, mbr, rover  # noqa: E402
from src.eval.edit_distance import levenshtein  # noqa: E402
from src.eval.eval_metrics import evaluate_set, length_bucket_report  # noqa: E402
from src.eval.normalize import norm_aggressive, norm_strip_ws  # noqa: E402
from src.make_submission import build  # noqa: E402
from src.postprocess import (ConfusionRules, PostprocessConfig, TrainLexicon,  # noqa: E402
                             apply_all, spacing_normalize)


@pytest.fixture
def tmpdata(tmp_path):
    ref = pd.DataFrame({
        "ID": ["a", "b", "c"],
        "Target": [
            "To all Christian People to whome these presents shall come",
            "Sealed and delivered in the presence of",
            "of February in the year of our Lord God one thousand six hundred sixty",
        ],
    })
    ref_p = tmp_path / "ref.csv"
    ref.to_csv(ref_p, index=False)
    return tmp_path, ref


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0


def test_perfect_prediction_scores_one(tmpdata):
    tmp, ref = tmpdata
    p = tmp / "pred.csv"
    ref.to_csv(p, index=False)
    r = evaluate_set(str(tmp / "ref.csv"), str(p), verbose=False)
    assert r["cer"] == pytest.approx(1.0)
    assert r["wer"] == pytest.approx(1.0)
    assert r["final_07_03"] == pytest.approx(1.0)
    assert r["final_05_05"] == pytest.approx(1.0)


def test_empty_predictions_score_zero(tmpdata):
    """A missing/empty prediction must be scored as fully wrong, not skipped."""
    tmp, ref = tmpdata
    p = tmp / "pred.csv"
    pd.DataFrame({"ID": ref["ID"], "Target": ["", "", ""]}).to_csv(p, index=False)
    r = evaluate_set(str(tmp / "ref.csv"), str(p), verbose=False)
    assert r["cer_error"] == pytest.approx(1.0)
    assert r["final_07_03"] == pytest.approx(0.0)


def test_missing_ids_are_penalised(tmpdata):
    tmp, ref = tmpdata
    p = tmp / "pred.csv"
    ref.head(1).to_csv(p, index=False)   # only 1 of 3 rows predicted
    r = evaluate_set(str(tmp / "ref.csv"), str(p), verbose=False)
    assert r["matched"] == 3          # all references still counted
    assert r["final_07_03"] < 0.5


def test_long_lines_carry_more_weight(tmpdata):
    """Core property of the metric: an error on the LONG line must cost more
    than the same error on a short line."""
    tmp, ref = tmpdata
    long_i = int(ref["Target"].str.len().idxmax())
    short_i = int(ref["Target"].str.len().idxmin())

    def score_with_broken(i):
        d = ref.copy()
        d.loc[i, "Target"] = "xxx"
        p = tmp / f"pred_{i}.csv"
        d.to_csv(p, index=False)
        return evaluate_set(str(tmp / "ref.csv"), str(p), verbose=False)["final_07_03"]

    assert score_with_broken(long_i) < score_with_broken(short_i)


def test_both_weightings_reported(tmpdata):
    tmp, ref = tmpdata
    d = ref.copy()
    d.loc[0, "Target"] = "To all Christian People to whome these present shall com"
    p = tmp / "pred.csv"
    d.to_csv(p, index=False)
    r = evaluate_set(str(tmp / "ref.csv"), str(p), verbose=False)
    assert "final_07_03" in r and "final_05_05" in r
    assert r["final_07_03"] != r["final_05_05"] or r["cer"] == r["wer"]


def test_bucket_report(tmpdata):
    tmp, ref = tmpdata
    p = tmp / "pred.csv"
    ref.to_csv(p, index=False)
    rep = length_bucket_report(str(tmp / "ref.csv"), str(p))
    assert "weight_share" in rep.columns
    assert rep["weight_share"].sum() == pytest.approx(1.0)


def test_normalizers():
    assert norm_strip_ws("  a   b ") == "a b"
    assert norm_aggressive("A, B!") == "a b"


def test_spacing_normalize():
    assert spacing_normalize("a  b ,c") == "a b,c"


def test_lexicon_confusion_rules(tmp_path):
    train = pd.DataFrame({
        "ID": ["1", "2", "3"],
        "Target": ["presence of the said", "presence of the said",
                   "presence of the said John"],
    })
    tc = tmp_path / "train.csv"
    train.to_csv(tc, index=False)
    lex = TrainLexicon.fit(str(tc))
    cfg = PostprocessConfig(min_word_freq=2)
    rules = ConfusionRules(lex, cfg)

    # "0f" -> "of": in-lexicon and frequent, so the 0->O... actually 0->o path
    fixed, applied = apply_all("presence 0f the said", lex, rules, cfg)
    assert "presence" in fixed
    # An already-correct string must be left alone.
    same, _ = apply_all("presence of the said", lex, rules, cfg)
    assert same == "presence of the said"


def test_lexicon_never_touches_known_words(tmp_path):
    train = pd.DataFrame({"ID": ["1"], "Target": ["one thousand six hundred"]})
    tc = tmp_path / "t.csv"
    train.to_csv(tc, index=False)
    lex = TrainLexicon.fit(str(tc))
    cfg = PostprocessConfig()
    out, applied = apply_all("one thousand six hundred", lex, ConfusionRules(lex, cfg), cfg)
    assert out == "one thousand six hundred"
    assert applied == []


def test_rover_majority_vote():
    cands = ["hello world", "hello world", "hallo world"]
    out = rover(cands, [1.0, 1.0, 1.0])
    assert out == "hello world"


def test_mbr_picks_a_real_candidate():
    cands = ["hello world", "hello world", "totally different"]
    out = mbr(cands, [1.0, 1.0, 1.0])
    assert out in cands
    assert out == "hello world"


def test_combine_end_to_end(tmp_path):
    a = pd.DataFrame({"ID": ["x", "y"], "rank": [0, 0],
                      "Target": ["the said John", "Sealed and delivered"],
                      "score": [-0.1, -0.2], "arm": "trocr"})
    b = pd.DataFrame({"ID": ["x", "y"], "rank": [0, 0],
                      "Target": ["the said John", "Sealed and deliuered"],
                      "score": [-0.1, -0.3], "arm": "vlm"})
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    a.to_csv(pa, index=False); b.to_csv(pb, index=False)
    out = combine([str(pa), str(pb)], method="mbr")
    assert set(out["ID"]) == {"x", "y"}
    assert (out["Target"].str.strip() != "").all()


def test_submission_guarantees(tmp_path):
    test = pd.DataFrame({"ID": ["a", "b", "c"]})
    tp = tmp_path / "test.csv"
    test.to_csv(tp, index=False)

    pred = pd.DataFrame({"ID": ["a", "b"], "Target": ["hello", ""]})
    pp = tmp_path / "pred.csv"
    pred.to_csv(pp, index=False)

    fb = pd.DataFrame({"ID": ["b"], "Target": ["fallback text"]})
    fp = tmp_path / "fb.csv"
    fb.to_csv(fp, index=False)

    train = pd.DataFrame({"ID": ["1", "2"], "Target": ["common line", "common line"]})
    trp = tmp_path / "train.csv"
    train.to_csv(trp, index=False)

    out = tmp_path / "sub.csv"
    sub = build(str(tp), str(pp), str(out), [str(fp)], str(trp))

    assert list(sub.columns) == ["ID", "Target"]
    assert len(sub) == 3
    assert sub["ID"].is_unique
    assert (sub["Target"].str.strip() != "").all()      # no empty predictions
    assert sub.loc[sub["ID"] == "b", "Target"].iloc[0] == "fallback text"
    assert sub.loc[sub["ID"] == "c", "Target"].iloc[0] == "common line"  # train fallback


def test_submission_rejects_newlines(tmp_path):
    test = pd.DataFrame({"ID": ["a"]})
    tp = tmp_path / "t.csv"; test.to_csv(tp, index=False)
    pred = pd.DataFrame({"ID": ["a"], "Target": ["line one\nline two"]})
    pp = tmp_path / "p.csv"; pred.to_csv(pp, index=False)
    out = tmp_path / "s.csv"
    sub = build(str(tp), str(pp), str(out), [], None)
    assert "\n" not in sub["Target"].iloc[0]        # sanitized, not crashed


def test_exclusion_list_is_intact():
    from src.data.exclusions import FLAGGED_IDS

    assert len(FLAGGED_IDS) == 21
    assert len(set(FLAGGED_IDS)) == 21
    assert "79tMUVyfIdy3GzkG" in FLAGGED_IDS


def test_pseudo_label_tripwire(tmp_path):
    """The manual-override tripwire must actually fire."""
    from src.pseudo_label import assert_no_manual_overrides

    assert_no_manual_overrides(str(tmp_path))       # clean: no raise
    (tmp_path / "manual_labels.csv").write_text("ID,Target\n")
    with pytest.raises(RuntimeError):
        assert_no_manual_overrides(str(tmp_path))


def test_stackmix_refuses_test_manifest(tmp_path):
    """StackMix must refuse a CSV without a Target column (i.e. a test manifest)."""
    from src.data.stackmix import StackMixConfig, build_crop_bank

    bad = pd.DataFrame({"ID": ["a", "b"]})
    p = tmp_path / "test_manifest.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError):
        build_crop_bank(str(p), str(tmp_path), str(tmp_path / "bank"), StackMixConfig())


def test_checkpoint_dir_naming_is_enforced(tmp_path):
    pytest.importorskip("torch")
    from src.utils.ckpt_manager import CheckpointManager

    CheckpointManager(str(tmp_path / "my_checkpoints"))       # ok
    with pytest.raises(ValueError):
        CheckpointManager(str(tmp_path / "weights"))          # missing "checkpoint"


def test_config_inheritance():
    from src.utils.common import load_config, parse_cli_overrides

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "configs", "trocr_large.yaml"))
    assert cfg["arm"] == "trocr"
    assert cfg["eval"]["weight_factor"] == 0.5          # inherited from base
    assert "checkpoint" in cfg["output"]["checkpoint_dir"]

    o = parse_cli_overrides(["train.lr=1e-5", "data.fold=2"])
    assert o["train"]["lr"] == 1e-5 and o["data"]["fold"] == 2


def test_all_configs_use_checkpoint_named_dirs():
    from src.utils.common import load_config

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cdir = os.path.join(root, "configs")
    for f in os.listdir(cdir):
        if not f.endswith(".yaml"):
            continue
        cfg = load_config(os.path.join(cdir, f))
        assert "checkpoint" in cfg["output"]["checkpoint_dir"], f
