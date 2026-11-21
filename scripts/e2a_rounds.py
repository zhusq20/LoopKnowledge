"""2A + 3C (zero training): per-round single-hop recall and two-hop composition.

For every fact (single hop) and every two-hop chain we read greedy answers at each exit
round. Prompt variants per chain:
  nested   : "The capital of the country of citizenship of X is"
  context  : first hop stated in context, then the nested cloze
  corrupt  : nested cloze with X replaced by another subject X' (same r1, different bridge
             and different answer); scored against the ORIGINAL answer = shortcut baseline
Outputs results/e2a/<model>/raw.json (all generations + match flags).

python scripts/e2a_rounds.py --model ouro-1.4b --rounds 1 2 3 4 5 6 7 8
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lk.data import facts_and_chains, load_cases, nested
from lk.evalrounds import FEWSHOT, eval_prompts
from lk.loopmodel import load

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--rounds", type=int, nargs="+", required=True)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--limit", type=int, default=0)
a = p.parse_args()

out = ROOT / "results" / "e2a" / a.model
out.mkdir(parents=True, exist_ok=True)
facts, chains = facts_and_chains(load_cases())
F = list(facts.values())
C = list(chains.values())
if a.limit:
    F, C = F[: a.limit], C[: a.limit]

# corrupted subject: same r1, different bridge and different answer
rng = random.Random(0)
by_r1 = {}
for c in chains.values():
    by_r1.setdefault(c["r1"], []).append(c)
for c in C:
    pool = [d for d in by_r1[c["r1"]] if d["bridge"] != c["bridge"] and not (set(d["aliases"]) & set(c["aliases"]))]
    d = rng.choice(pool) if pool else None
    c["corrupt_s1"] = d["s1"] if d else None
    c["corrupt_cloze"] = nested(c["t2"], c["r1_label"], d["s1"]) if d else None

m = load(a.model)
log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
t0 = time.time()
rf = eval_prompts(m, [f["cloze"] for f in F], {"correct": [f["aliases"] for f in F]}, a.rounds, FEWSHOT, batch=a.batch, log=lambda s: log("facts " + s))
rn = eval_prompts(m, [c["cloze"] for c in C], {"correct": [c["aliases"] for c in C]}, a.rounds, FEWSHOT, batch=a.batch, log=lambda s: log("nested " + s))
rc = eval_prompts(m, [c["context_cloze"] for c in C], {"correct": [c["aliases"] for c in C]}, a.rounds, FEWSHOT, batch=a.batch, log=lambda s: log("context " + s))
Ck = [c for c in C if c["corrupt_cloze"]]
rx = eval_prompts(m, [c["corrupt_cloze"] for c in Ck], {"orig_answer": [c["aliases"] for c in Ck]}, a.rounds, FEWSHOT, batch=a.batch, log=lambda s: log("corrupt " + s))

raw = {
    "model": a.model,
    "rounds": a.rounds,
    "format": "fewshot",
    "seconds": time.time() - t0,
    "facts": [{**f, **{f"r{r}": {"gen": rf[r]["gen"][i], "ok": rf[r]["correct"][i]} for r in a.rounds}} for i, f in enumerate(F)],
    "chains": [],
}
ki = {c["key"]: j for j, c in enumerate(Ck)}
for i, c in enumerate(C):
    row = dict(c)
    for r in a.rounds:
        row[f"r{r}"] = {
            "nested_gen": rn[r]["gen"][i],
            "nested_ok": rn[r]["correct"][i],
            "context_gen": rc[r]["gen"][i],
            "context_ok": rc[r]["correct"][i],
        }
        if c["key"] in ki:
            j = ki[c["key"]]
            row[f"r{r}"]["corrupt_gen"] = rx[r]["gen"][j]
            row[f"r{r}"]["corrupt_ok"] = rx[r]["orig_answer"][j]
    raw["chains"].append(row)
json.dump(raw, open(out / "raw.json", "w"))
log(f"done in {time.time() - t0:.0f}s -> {out / 'raw.json'}")
