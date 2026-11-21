"""2E + 1C: write new atomic facts by fine-tuning, test direct composition by round, and
decompose the weight change by layer part and (for looped models) by round.

World (seeded): 128 chains  person P --works for--> company C --CEO--> person Q, all
fictional. Only atomic facts are trained, 3 phrasings each; a 4th phrasing is held out.
The two-hop nested cloze "The chief executive officer of the employer of P is" is never
trained (double-new direct composition). A context control states the first hop.

Trainable = MLP down_proj deltas of the chosen layers (default: all), shared over rounds.
Arms: --sup final (loss on the final round only) or --sup all (uniform over rounds).
After training: per-round reads with the full delta, then the delta restricted to each
layer part (thirds; Huginn: prelude / core / coda) and, for loops, to single rounds.

python scripts/e2e_newfacts.py --model ouro-1.4b --tag ouro-all-final
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

from lk.data import alias_list, facts_and_chains, load_cases
from lk.edit import build_examples, nll
from lk.evalrounds import FEWSHOT, eval_prompts
from lk.loopmodel import Edit, load

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--T", type=int, default=None)
p.add_argument("--layers", nargs="*", default=None, help="default: every MLP")
p.add_argument("--sup", default="final", choices=["final", "all"])
p.add_argument("--lr", type=float, default=None)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--stop-acc", type=float, default=0.95, help="stop when train-phrasing greedy acc reaches this")
p.add_argument("--n", type=int, default=128)
p.add_argument("--read-rounds", type=int, nargs="+", default=None)
p.add_argument("--tag", required=True)
a = p.parse_args()

m = load(a.model)
T = a.T or {"ouro": 4, "huginn": 32, "ff": 1}[m.kind]
RR = a.read_rounds or ({"ouro": [1, 2, 3, 4], "huginn": [4, 8, 16, 32], "ff": [1]}[m.kind])
lr = a.lr or {"ouro": 1e-4, "huginn": 1e-5, "ff": 1e-4}[m.kind]
out = ROOT / "results" / "e2e" / a.model / a.tag
out.mkdir(parents=True, exist_ok=True)
log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
json.dump(vars(a), open(out / "args.json", "w"), indent=1)

# ------------------------------------------------------------------ world
rng = random.Random(7)
SYL = "ka lo mi re tu sa vo ne di pa zu ri mo ta le gu fi ha no be ki ru so ma".split()


def name(k):
    return "".join(rng.choice(SYL) for _ in range(k)).capitalize()


used = set()


def fresh(fn):
    while True:
        x = fn()
        if x not in used:
            used.add(x)
            return x


P = [fresh(lambda: name(2) + " " + name(3)) for _ in range(a.n)]
C = [fresh(lambda: name(3) + rng.choice([" Industries", " Systems", " Labs", " Group"])) for _ in range(a.n)]
Q = [fresh(lambda: name(2) + " " + name(2)) for _ in range(a.n)]
R1 = ["{p} works for", "{p} is employed by", "The employer of {p} is", "{p} has a job at"]  # last = held out
R2 = ["The chief executive officer of {c} is", "{c} is led by its CEO,", "The CEO of {c} is", "The person who runs {c} as CEO is"]
train_items, held, nested_items, ctx_items = [], [], [], []
for i in range(a.n):
    for t in R1[:3]:
        train_items.append((t.format(p=P[i]), C[i]))
    for t in R2[:3]:
        train_items.append((t.format(c=C[i]), Q[i]))
    held.append(("hop1", R1[3].format(p=P[i]), C[i]))
    held.append(("hop2", R2[3].format(c=C[i]), Q[i]))
    nested_items.append((f"The chief executive officer of the employer of {P[i]} is", Q[i]))
    ctx_items.append((f"{P[i]} works for {C[i]}. The chief executive officer of the employer of {P[i]} is", Q[i]))
json.dump({"P": P, "C": C, "Q": Q, "R1": R1, "R2": R2}, open(out / "world.json", "w"), indent=1)

# known-fact retention probe
facts, _ = facts_and_chains(load_cases())
pool = [f for f in facts.values() if f["split"] == "test"]
random.Random(3).shuffle(pool)
pool = pool[:200]
pre = eval_prompts(m, [f["cloze"] for f in pool], {"ok": [f["aliases"] for f in pool]}, [T], FEWSHOT, batch=64)
known = [f for f, ok in zip(pool, pre[T]["ok"]) if ok][:64]


def evaluate(edit, rounds):
    res = {}
    sets = {
        "train": [(x, y) for x, y in train_items[:: 6 * max(1, a.n // 32)]],
        "held_hop1": [(x, y) for k, x, y in held if k == "hop1"],
        "held_hop2": [(x, y) for k, x, y in held if k == "hop2"],
        "nested": nested_items,
        "context": ctx_items,
    }
    for nm, items in sets.items():
        r = eval_prompts(m, [x for x, _ in items], {"ok": [alias_list(y, []) for _, y in items]}, rounds, FEWSHOT, batch=64, edit=edit)
        res[nm] = {str(rr): float(np.mean(r[rr]["ok"])) for rr in rounds}
    r = eval_prompts(m, [f["cloze"] for f in known], {"ok": [f["aliases"] for f in known]}, [T], FEWSHOT, batch=64, edit=edit)
    res["known_retained"] = float(np.mean(r[T]["ok"]))
    return res


base = evaluate(None, RR)
log(f"before: {base}")

# ------------------------------------------------------------------ train
keys = m.mlp_keys() if not a.layers else [(x.split(":")[0], int(x.split(":")[1])) if ":" in x else int(x) for x in a.layers]
deltas = {k: torch.zeros_like(m.down_proj(k), dtype=torch.float32, requires_grad=True) for k in keys}
edit = Edit(deltas, None)
opt = torch.optim.Adam(deltas.values(), lr=lr)
ex = build_examples(m, train_items, (FEWSHOT, ""))
sup = [T] if a.sup == "final" else list(range(1, T + 1))
hist = []
step = 0
for ep in range(a.epochs):
    random.Random(ep).shuffle(ex)
    losses = []
    for b in range(0, len(ex), a.bs):
        with torch.enable_grad():
            opt.zero_grad(set_to_none=True)
            loss, _ = nll(m, ex[b : b + a.bs], edit, T, sup)
            loss.backward()
            opt.step()
        losses.append(float(loss))
        step += 1
    frozen = Edit({k: d.detach() for k, d in deltas.items()}, None)
    tr = eval_prompts(m, [x for x, _ in train_items], {"ok": [alias_list(y, []) for _, y in train_items]}, [T], FEWSHOT, batch=64, edit=frozen)
    acc = float(np.mean(tr[T]["ok"]))
    hist.append({"epoch": ep, "step": step, "loss": float(np.mean(losses)), "train_acc": acc})
    log(f"epoch {ep} loss {np.mean(losses):.4f} train acc {acc:.3f}")
    if acc >= a.stop_acc:
        break
D = {k: d.detach() for k, d in deltas.items()}
full = evaluate(Edit(D, None), RR)
log(f"after: {full}")

# ------------------------------------------------------------------ decompose (1C)
dec = {}
if m.kind == "huginn":
    parts = {"prelude": [k for k in keys if isinstance(k, tuple) and k[0] == "prelude"], "core": [k for k in keys if not isinstance(k, tuple)], "coda": [k for k in keys if isinstance(k, tuple) and k[0] == "coda"]}
else:
    ints = [k for k in keys if not isinstance(k, tuple)]
    n3 = max(1, len(ints) // 3)
    parts = {"lower": ints[:n3], "middle": ints[n3 : 2 * n3], "upper": ints[2 * n3 :]} if len(ints) >= 3 else {}
for pn, ks in parts.items():
    if ks:
        dec[f"part:{pn}"] = evaluate(Edit({k: D[k] for k in ks}, None), [T])
        dec[f"drop:{pn}"] = evaluate(Edit({k: D[k] for k in keys if k not in ks}, None), [T])
if m.kind == "ouro":
    for r in range(1, T + 1):
        dec[f"round:{r}"] = evaluate(Edit(D, {r}), [T])
        dec[f"allbut:{r}"] = evaluate(Edit(D, set(range(1, T + 1)) - {r}), [T])
norms = {str(k): float(D[k].norm()) for k in keys}
json.dump({"before": base, "after": full, "decompose": dec, "history": hist, "delta_norms": norms, "n_known_probe": len(known)}, open(out / "summary.json", "w"), indent=1)
if m.kind != "huginn":
    torch.save({str(k): v.to(torch.bfloat16).cpu() for k, v in D.items()}, out / "delta.pt")
log("done")
