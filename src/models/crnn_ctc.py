"""
Arm C - CRNN/CTC trained from scratch on the provided data only.

Why a third arm: the TrOCR and VLM arms are both autoregressive transformers and
make correlated mistakes (notably hallucinated continuations on faded lines). A
CTC model cannot hallucinate beyond the visual evidence, so it is a genuinely
diverse ensemble member and a strong voter in the character-level fusion.

Licensing: this architecture is defined here in-repo (no pretrained weights are
downloaded), so it is unambiguously inside the "open-source tools only" rule and
carries no third-party weights license at all.

The charset is derived EXCLUSIVELY from the training transcriptions.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
class CharTokenizer:
    """CTC charset built from train transcriptions only. Index 0 = blank."""

    BLANK = "<blank>"
    UNK = "<unk>"

    def __init__(self, chars: List[str]):
        self.itos = [self.BLANK, self.UNK] + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_texts(cls, texts, min_count: int = 1) -> "CharTokenizer":
        from collections import Counter

        cnt = Counter()
        for t in texts:
            cnt.update(str(t))
        chars = sorted([c for c, n in cnt.items() if n >= min_count])
        return cls(chars)

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, 1) for c in str(text)]

    def decode_greedy(self, ids: List[int]) -> str:
        out, prev = [], -1
        for i in ids:
            if i != prev and i != 0:
                out.append(self.itos[i] if i < len(self.itos) else "")
            prev = i
        return "".join(ch for ch in out if ch not in (self.BLANK, self.UNK))

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path) as f:
            itos = json.load(f)["itos"]
        tk = cls([])
        tk.itos = itos
        tk.stoi = {c: i for i, c in enumerate(itos)}
        return tk


# --------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, pool=(2, 2)):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(pool) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.body(x))


class CRNN(nn.Module):
    """CNN feature extractor -> width-wise sequence -> BiLSTM -> CTC logits.

    Height is fully collapsed; width is downsampled by 4 so a 1024px canvas
    yields 256 timesteps, comfortably above the 120-char max target length.
    """

    def __init__(self, n_classes: int, in_ch: int = 1, hidden: int = 384,
                 layers: int = 3, dropout: float = 0.15):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(in_ch, 64, (2, 2)),
            ConvBlock(64, 128, (2, 2)),
            ConvBlock(128, 256, (2, 1)),
            ConvBlock(256, 384, (2, 1)),
            nn.Conv2d(384, 384, 3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(384, hidden, num_layers=layers, bidirectional=True,
                           batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> log-probs (T, B, n_classes) for nn.CTCLoss."""
        f = self.cnn(x)               # (B, C, 1, W')
        f = f.squeeze(2).permute(0, 2, 1)   # (B, W', C)
        f, _ = self.rnn(f)
        logits = self.head(self.drop(f))    # (B, T, n_classes)
        return logits.log_softmax(-1).permute(1, 0, 2)

    @torch.no_grad()
    def logprobs_btc(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) log-probs, convenient for beam search / ensembling."""
        return self.forward(x).permute(1, 0, 2)


# --------------------------------------------------------------------------
def ctc_beam_search(logprobs: torch.Tensor, tokenizer: CharTokenizer,
                    beam_width: int = 10, top_k: int = 1) -> List[Tuple[str, float]]:
    """Vanilla prefix beam search over a single sample's (T, C) log-probs.

    Returns up to `top_k` (text, logprob) candidates - the extra candidates feed
    the ensemble's minimum-weighted-CER selection.
    """
    import math

    T, C = logprobs.shape
    beams: Dict[str, Tuple[float, float]] = {"": (0.0, -math.inf)}  # prefix -> (p_blank, p_nonblank)
    lp = logprobs.detach().float().cpu().numpy()

    def logsumexp(a, b):
        if a == -math.inf:
            return b
        if b == -math.inf:
            return a
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    for t in range(T):
        cand = min(C, beam_width * 4)
        top = lp[t].argsort()[-cand:]
        nxt: Dict[str, Tuple[float, float]] = {}
        for prefix, (pb, pnb) in beams.items():
            for c in top:
                p = float(lp[t, c])
                if c == 0:  # blank
                    b, nb = nxt.get(prefix, (-math.inf, -math.inf))
                    nxt[prefix] = (logsumexp(b, logsumexp(pb, pnb) + p), nb)
                    continue
                ch = tokenizer.itos[c]
                last = prefix[-1] if prefix else None
                if ch == last:
                    # repeat without blank extends the same char
                    b, nb = nxt.get(prefix, (-math.inf, -math.inf))
                    nxt[prefix] = (b, logsumexp(nb, pnb + p))
                    np_ = prefix + ch
                    b2, nb2 = nxt.get(np_, (-math.inf, -math.inf))
                    nxt[np_] = (b2, logsumexp(nb2, pb + p))
                else:
                    np_ = prefix + ch
                    b2, nb2 = nxt.get(np_, (-math.inf, -math.inf))
                    nxt[np_] = (b2, logsumexp(nb2, logsumexp(pb, pnb) + p))
        scored = sorted(nxt.items(),
                        key=lambda kv: -logsumexp(kv[1][0], kv[1][1]))[:beam_width]
        beams = dict(scored)

    ranked = sorted(beams.items(), key=lambda kv: -logsumexp(kv[1][0], kv[1][1]))
    return [(p, logsumexp(v[0], v[1])) for p, v in ranked[:top_k]]
