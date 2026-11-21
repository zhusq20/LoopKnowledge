"""Phase-0 check: our round-by-round forward equals the official forward, padding is
harmless, and per-round edits behave as stated (all rounds == editing shared weights).

Run: CUDA_VISIBLE_DEVICES=5 python tests/check_forward.py ouro-1.4b
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from lk.loopmodel import Edit, load, pad_right, round_logits_last

torch.backends.cuda.matmul.allow_tf32 = False
name = sys.argv[1]
dtype = getattr(torch, sys.argv[2]) if len(sys.argv) > 2 else torch.bfloat16
m = load(name, dtype=dtype)
texts = ["The capital of France is", "Fernando Santos is a citizen of the country of", "Paris"]
seqs = [([m.bos] if m.bos is not None else []) + m.tok(t, add_special_tokens=False).input_ids for t in texts]
T = {"ouro": 4, "huginn": 8, "ff": 1}[m.kind]
res = {}

with torch.no_grad():
    # 1) official forward, one sequence at a time
    off = []
    for s in seqs:
        ids = torch.tensor([s], device=m.device)
        if m.kind == "ff":
            off.append(m.lm(input_ids=ids).logits[0, -1:].float())
        elif m.kind == "ouro":
            _, hs, _ = m.lm.model(input_ids=ids, use_cache=False)
            off.append(torch.stack([m.lm.lm_head(h[0, -1]).float() for h in hs[:T]]))
        else:
            rows = []
            for t in range(1, T + 1):
                st = m.init[: len(s)][None].clone()
                rows.append(m.lm(input_ids=ids, input_states=st, num_steps=t).logits[0, -1].float())
            off.append(torch.stack(rows))
    off = torch.stack(off, 1)  # [T, B, V]
    # 2) ours, padded batch
    ours = round_logits_last(m, seqs, T)
    d = (ours - off).abs().amax(-1)
    agree = (ours.argmax(-1) == off.argmax(-1)).float().mean().item()
    res["max_abs_logit_diff_per_round"] = d.amax(-1).tolist()
    res["argmax_agreement"] = agree
    # 3) per-round edit: all rounds == shared edit; no rounds == base
    key = m.mlp_keys()[len(m.mlp_keys()) // 2]
    W = m.down_proj(key)
    dW = torch.randn_like(W, dtype=torch.float32) * W.float().std() * 0.5
    e_all = round_logits_last(m, seqs, T, edit=Edit({key: dW}, None))
    e_rounds = round_logits_last(m, seqs, T, edit=Edit({key: dW}, set(range(1, T + 1))))
    e_none = round_logits_last(m, seqs, T, edit=Edit({key: dW}, set()))
    W0 = W.data.clone()
    W.data = (W0.float() + dW).to(W0.dtype)
    shared = round_logits_last(m, seqs, T)
    W.data = W0
    res["edit_all_vs_rounds_all"] = (e_all - e_rounds).abs().max().item()
    res["edit_none_vs_base"] = (e_none - ours).abs().max().item()
    res["edit_delta_vs_shared_weight_maxabs"] = (e_all - shared).abs().max().item()
    res["edit_delta_vs_shared_weight_argmax_agree"] = (e_all.argmax(-1) == shared.argmax(-1)).float().mean().item()
    res["edit_changes_output"] = (e_all - ours).abs().max().item()
print(res)
