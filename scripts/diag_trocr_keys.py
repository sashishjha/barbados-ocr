"""Compare golden TrOCR state_dict keys (saved under transformers 5.x) with the keys the
current transformers builds, grouped by pattern (layer numbers -> N). CPU only."""
import re
import sys

import torch
import yaml

sys.path.insert(0, ".")
from src.train_trocr import build_model  # noqa: E402

gold = sys.argv[1]
cfg = yaml.safe_load(open("configs/trocr_large.yaml"))
cfg.setdefault("model", {})["attn_implementation"] = "eager"
model, _ = build_model(cfg)
want = set(model.state_dict())
have = set(torch.load(f"{gold}/model.pt", map_location="cpu", weights_only=False))


def pat(keys):
    return sorted({re.sub(r"\.\d+\.", ".N.", k) for k in keys})


print("missing (model wants):"); print("\n".join(pat(want - have)))
print("\nunexpected (file has):"); print("\n".join(pat(have - want)))
print(f"\nshared keys: {len(want & have)}  missing: {len(want - have)}  unexpected: {len(have - want)}")

# Verify the converter: strict load must succeed and weights must match the file.
from src.utils.ckpt_manager import _vit_keys_v5_to_v4  # noqa: E402
sd = _vit_keys_v5_to_v4(torch.load(f"{gold}/model.pt", map_location="cpu", weights_only=False))
model.load_state_dict(sd, strict=True)
k = "encoder.encoder.layer.5.attention.attention.query.weight"
assert torch.equal(model.state_dict()[k], sd[k])
print("STRICT LOAD OK after conversion")
