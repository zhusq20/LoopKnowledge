"""3A, parametric chains (zero training): per-round accuracy for k = 1, 2, 3 known-fact chains.

k=1 and k=2 come from results/e2a/<model>/raw.json. This script adds k=3 nested clozes
("The capital of the country of citizenship of the spouse of X is") and writes
results/e3a/<model>/param_chains.json with, for each k, accuracy by round on chains whose
component single hops are all answered at the final round ("known" chains). Shortcut
control for k=3: X replaced by another subject with the same first relation.

python scripts/e3a_param_chains.py --model ouro-1.4b --rounds 1 2 3 4 5 6 7 8
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from lk.data import load_cases, three_hop_chains
from lk.evalrounds import FEWSHOT, eval_prompts
from lk.loopmodel import load

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--rounds", type=int, nargs="+", required=True)
p.add_argument("--final", type=int, default=None)
p.add_argument("--batch", type=int, default=128)
a = p.parse_args()
final = a.final or (4 if a.model.startswith("ouro") else 32)

raw = json.load(open(ROOT / "results" / "e2a" / a.model / "raw.json"))
F = {f["key"]: f for f in raw["facts"]}
t3 = list(three_hop_chains(load_cases()).values())
# keep chains whose hops are all in the e2a fact table
t3 = [c for c in t3 if all(h in F for h in c["hops"])]
known3 = [c for c in t3 if all(F[h][f"r{final}"]["ok"] for h in c["hops"])]
rng = random.Random(0)
subjects = sorted({c["hops"][0].split("||")[0] for c in t3})
for c in known3:
    s1 = c["hops"][0].split("||")[0]
    other = [x for x in t3 if x["hops"][0].split("||")[1] == c["hops"][0].split("||")[1] and x["hops"][0].split("||")[0] != s1 and not (set(x["aliases"]) & set(c["aliases"]))]
    c["corrupt_cloze"] = c["cloze"].replace(s1, rng.choice(other)["hops"][0].split("||")[0], 1) if other else None

m = load(a.model)
res = eval_prompts(m, [c["cloze"] for c in known3], {"ok": [c["aliases"] for c in known3]}, a.rounds, FEWSHOT, batch=a.batch, log=print)
ck = [c for c in known3 if c["corrupt_cloze"]]
resx = eval_prompts(m, [c["corrupt_cloze"] for c in ck], {"ok": [c["aliases"] for c in ck]}, a.rounds, FEWSHOT, batch=a.batch, log=print)
cx = {c["key"]: i for i, c in enumerate(ck)}

out = {"model": a.model, "rounds": a.rounds, "final": final, "k": {}}
facts_known = [f for f in raw["facts"] if f[f"r{final}"]["ok"]]
out["k"]["1"] = {"n": len(facts_known), "acc": {r: float(np.mean([f[f"r{r}"]["ok"] for f in facts_known])) for r in raw["rounds"] if r in a.rounds}}
ch2 = [c for c in raw["chains"] if F[c["hop1"]][f"r{final}"]["ok"] and F[c["hop2"]][f"r{final}"]["ok"]]
ch2n = [c for c in ch2 if not c[f"r{final}"].get("corrupt_ok", False)]
out["k"]["2"] = {
    "n": len(ch2),
    "acc": {r: float(np.mean([c[f"r{r}"]["nested_ok"] for c in ch2])) for r in a.rounds if f"r{r}" in ch2[0]},
    "n_no_shortcut": len(ch2n),
    "acc_no_shortcut": {r: float(np.mean([c[f"r{r}"]["nested_ok"] for c in ch2n])) for r in a.rounds if f"r{r}" in ch2n[0]},
}
sc = [i for i, c in enumerate(known3) if c["corrupt_cloze"] and resx[final]["ok"][cx[c["key"]]]]
keep = [i for i in range(len(known3)) if i not in set(sc)]
out["k"]["3"] = {
    "n": len(known3),
    "acc": {r: float(np.mean(res[r]["ok"])) for r in a.rounds},
    "n_no_shortcut": len(keep),
    "acc_no_shortcut": {r: float(np.mean([res[r]["ok"][i] for i in keep])) for r in a.rounds},
    "corrupt_acc": {r: float(np.mean(resx[r]["ok"])) for r in a.rounds},
}
out["k3_rows"] = [{"cloze": c["cloze"], "aliases": c["aliases"][:3], **{f"r{r}": res[r]["gen"][i] for r in a.rounds}} for i, c in enumerate(known3)]
d = ROOT / "results" / "e3a" / a.model
d.mkdir(parents=True, exist_ok=True)
json.dump(out, open(d / "param_chains.json", "w"), indent=1)
for k, v in out["k"].items():
    print(k, v["n"], {r: round(x, 3) for r, x in v["acc"].items()})
    if "acc_no_shortcut" in v:
        print("  no-shortcut", v["n_no_shortcut"], {r: round(x, 3) for r, x in v["acc_no_shortcut"].items()})
