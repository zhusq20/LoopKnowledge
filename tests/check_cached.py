"""KV-cached greedy decoding must reproduce full-recompute greedy decoding.

CUDA_VISIBLE_DEVICES=5 python tests/check_cached.py ouro-1.4b 1 2 3 4
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from lk.data import facts_and_chains, load_cases
from lk.evalrounds import FEWSHOT, encode
from lk.loopmodel import Edit, greedy_at_round, greedy_cached, load

name, rounds = sys.argv[1], [int(x) for x in sys.argv[2:]]
m = load(name, dtype=getattr(torch, __import__("os").environ.get("DT", "bfloat16")))
facts, chains = facts_and_chains(load_cases())
texts = [f["cloze"] for f in list(facts.values())[:48]] + [c["cloze"] for c in list(chains.values())[:16]]
enc = [encode(m, t, FEWSHOT) for t in texts]
key = m.mlp_keys()[len(m.mlp_keys()) // 3]
W = m.down_proj(key)
g = torch.Generator(device=m.device).manual_seed(0)
dW = torch.randn(W.shape, generator=g, device=m.device) * 2e-4
for edit in [None, Edit({key: dW}, None)]:
    for r in rounds:
        t0 = time.time()
        a = greedy_at_round(m, enc, r, 10, None, edit)
        t1 = time.time()
        b = greedy_cached(m, enc, r, 10, edit)
        t2 = time.time()
        same = sum(x == y for x, y in zip(a, b))
        first = sum(x[:1] == y[:1] for x, y in zip(a, b))
        print(f"edit={edit is not None} r={r}: identical {same}/{len(a)}, first token {first}/{len(a)}, recompute {t1 - t0:.1f}s cached {t2 - t1:.1f}s", flush=True)
        for x, y in zip(a, b):
            if x != y:
                print("   diff:", repr(m.tok.decode(x)), "|", repr(m.tok.decode(y)))
                break
