"""Summaries for 1A traces: python scripts/analyze_e1a.py ouro-1.4b trace_T4.json

Indirect effect (IE) of a restoration = p_restored - p_corrupt, averaged over facts.
Knockout effect = (p_ko - p_clean) / p_clean, averaged over facts.
Prints per-round summaries (max and mean over the layers of that round) and the
argmax site, and writes summary.json with the full mean maps.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
model, fname = sys.argv[1], sys.argv[2]
d = json.load(open(ROOT / "results" / "e1a" / model / fname))
sites = [tuple(s) if not isinstance(s[1], list) else (s[0], tuple(s[1])) for s in d["sites"]]
items = d["items"]
pc = np.array([it["p_clean"] for it in items])
pk = np.array([it["p_corrupt"] for it in items])
S = {"model": model, "file": fname, "n": len(items), "p_clean": float(pc.mean()), "p_corrupt": float(pk.mean()), "maps": {}}


def group_of(s):
    r, li = s
    if isinstance(li, tuple):
        return li[0]
    return f"round{r}"


groups = []
for s in sites:
    g = group_of(s)
    if g not in groups:
        groups.append(g)

L = [f"# 1A — {model} ({fname})", "", f"facts n={len(items)}, mean p_clean={pc.mean():.3f}, mean p_corrupt={pk.mean():.3f}", ""]
for key in ["hid_subj", "hid_last", "mlp_subj", "attn_subj", "mlp_last", "attn_last", "ko_attn", "ko_mlp"]:
    M = np.array([it[key] for it in items])
    if key.startswith("ko"):
        E = (M - pc[:, None]) / pc[:, None]
    else:
        E = M - pk[:, None]
    mean = E.mean(0)
    S["maps"][key] = mean.tolist()
    row = []
    for g in groups:
        idx = [i for i, s in enumerate(sites) if group_of(s) == g]
        v = mean[idx]
        best = idx[int(np.argmin(v) if key.startswith("ko") else np.argmax(v))]
        row.append(f"{g}: {'min' if key.startswith('ko') else 'max'} {v.min() if key.startswith('ko') else v.max():+.3f} @ {sites[best][1]}, mean {v.mean():+.3f}")
    L.append(f"**{key}**  " + " | ".join(row))
    L.append("")
json.dump(S, open(ROOT / "results" / "e1a" / model / f"summary_{fname}", "w"), indent=1)
open(ROOT / "results" / "e1a" / model / f"summary_{Path(fname).stem}.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
