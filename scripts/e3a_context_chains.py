"""3A, in-context chains (zero training): accuracy by chain length k and exit round.

Each prompt states 4 disjoint chains of 7 people ("The boss of A is B."), all facts
shuffled, then asks "The boss of the boss of ... of A is" (k nested lookups, k = 1..6).
Scoring: candidate set = every name in the prompt (28 single-token names); the answer is
the candidate with the highest next-token logit at the final position. One forward pass
reads every round. Chance = 1/28. No chain of thought.

python scripts/e3a_context_chains.py --model ouro-1.4b --rounds 1 2 3 4 5 6 7 8
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch

from lk.loopmodel import load, round_logits_last

NAMES = "Alice Bob Carol Dave Emma Frank Grace Henry Jack Kate Leo Mary Nick Olivia Paul Quinn Rose Sam Victor Adam Ben Chris Dan Eric Fred George Ivan Jane Kevin Laura Mike Oscar Peter Ryan Sara Tom Will Anna Bill Clara Gary Ian Jill Karl Lucy Mark".split()
REL = ["boss", "neighbor", "teacher"]

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--rounds", type=int, nargs="+", required=True)
p.add_argument("--n", type=int, default=300, help="prompts per k")
p.add_argument("--kmax", type=int, default=6)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--style", default="nested", choices=["nested", "possessive"])
p.add_argument("--chains", type=int, default=4)
p.add_argument("--clen", type=int, default=7, help="people per chain")
p.add_argument("--demo", action="store_true", help="prepend one solved example (other names, same k)")
p.add_argument("--tag", default="")
a = p.parse_args()

torch.set_grad_enabled(False)
m = load(a.model)
names = [n for n in NAMES if len(m.tok(" " + n, add_special_tokens=False).input_ids) == 1]
tid = {n: m.tok(" " + n, add_special_tokens=False).input_ids[0] for n in names}
log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
log(f"{len(names)} single-token names")
rng = random.Random(0)


def block(k, rel, pool):
    C, Lc = a.chains, a.clen
    ppl = rng.sample(pool, C * Lc)
    chains = [ppl[i * Lc : (i + 1) * Lc] for i in range(C)]
    if a.style == "nested":
        facts = [f"The {rel} of {c[j]} is {c[j + 1]}." for c in chains for j in range(Lc - 1)]
    else:
        facts = [f"{c[j]}'s {rel} is {c[j + 1]}." for c in chains for j in range(Lc - 1)]
    rng.shuffle(facts)
    c = chains[0]
    start = rng.randrange(0, Lc - k)
    if a.style == "nested":
        q = "{}"
        for _ in range(k):
            q = q.format(f"the {rel} of {{}}")
        q = q.format(c[start])
        q = q[0].upper() + q[1:] + " is"
    else:
        q = c[start] + "".join(f"'s {rel}" for _ in range(k)) + " is"
    return " ".join(facts) + "\n" + q, ppl, c, start


def make(k):
    rel = rng.choice(REL)
    text, ppl, c, start = block(k, rel, names)
    if a.demo:
        rest = [n for n in names if n not in ppl]
        dt, dp, dc, ds = block(k, rel, rest) if len(rest) >= a.chains * a.clen else block(k, rel, names)
        text = dt + " " + dc[ds + k] + ".\n\n" + text
    return {"k": k, "rel": rel, "text": text, "gold": c[start + k], "cands": ppl, "start": c[start], "chain": c[start : start + k + 1]}


items = [make(k) for k in range(1, a.kmax + 1) for _ in range(a.n)]
T = max(a.rounds)
acc = {}
t0 = time.time()
rows = []
for b in range(0, len(items), a.batch):
    chunk = items[b : b + a.batch]
    seqs = [([m.bos] if m.bos is not None else []) + m.tok(x["text"], add_special_tokens=False).input_ids for x in chunk]
    lg = round_logits_last(m, seqs, T, rounds_keep=set(a.rounds))  # [len(rounds), B, V]
    for i, x in enumerate(chunk):
        ci = torch.tensor([tid[n] for n in x["cands"]], device=m.device)
        sc = lg[:, i, ci]  # [R, 28]
        pred = sc.argmax(-1).tolist()
        g = x["cands"].index(x["gold"])
        s = x["cands"].index(x["start"])
        hop = [x["chain"].index(x["cands"][pp]) if x["cands"][pp] in x["chain"] else -1 for pp in pred]
        rows.append({"k": x["k"], "ok": [int(pp == g) for pp in pred], "pred_hop": hop, "pred": [x["cands"][pp] for pp in pred]})
log(f"done {len(items)} prompts in {time.time() - t0:.0f}s")
res = {}
for k in range(1, a.kmax + 1):
    rk = [r for r in rows if r["k"] == k]
    okm = np.array([r["ok"] for r in rk])
    res[k] = {str(r): float(okm[:, j].mean()) for j, r in enumerate(a.rounds)}
out = ROOT / "results" / "e3a" / a.model
out.mkdir(parents=True, exist_ok=True)
json.dump({"model": a.model, "rounds": a.rounds, "n_per_k": a.n, "chance": 1 / (a.chains * a.clen), "args": vars(a), "acc": res, "rows": rows, "example": items[a.n * 2]["text"]}, open(out / f"context_chains{a.tag}.json", "w"))
print("k  " + " ".join(f"r{r:>3}" for r in a.rounds))
for k, v in res.items():
    rk = [r for r in rows if r["k"] == k]
    hop1 = np.mean([r["pred_hop"][a.rounds.index(T)] == 1 for r in rk])
    print(f"{k}  " + " ".join(f"{v[str(r)]:.2f}" for r in a.rounds) + f"   | final-round pred = hop-1 bridge: {hop1:.2f}")
