"""Phase 0: choose the cloze format per model from DEV single-hop coverage only.

python scripts/p0_format.py --model ouro-1.4b --rounds 1 2 3 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lk.data import facts_and_chains, load_cases
from lk.evalrounds import FORMATS, eval_prompts
from lk.loopmodel import load

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--rounds", type=int, nargs="+", required=True)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--limit", type=int, default=0)
a = p.parse_args()

out = ROOT / "results" / "p0-format"
out.mkdir(parents=True, exist_ok=True)
m = load(a.model)
facts, _ = facts_and_chains(load_cases())
dev = [f for f in facts.values() if f["split"] == "dev"]
if a.limit:
    dev = dev[: a.limit]
summary = {"model": a.model, "n_dev_facts": len(dev), "formats": {}}
for fmt, prefix in FORMATS.items():
    t0 = time.time()
    res = eval_prompts(m, [f["cloze"] for f in dev], {"correct": [f["aliases"] for f in dev]}, a.rounds, prefix, batch=a.batch, log=lambda s: print(fmt, s, flush=True))
    summary["formats"][fmt] = {
        "acc_by_round": {r: sum(v["correct"]) / len(dev) for r, v in res.items()},
        "seconds": time.time() - t0,
        "examples": [{"cloze": dev[i]["cloze"], "gold": dev[i]["aliases"][0], "gen": {r: res[r]["gen"][i] for r in res}} for i in range(0, len(dev), max(1, len(dev) // 12))],
    }
best = max(summary["formats"], key=lambda k: summary["formats"][k]["acc_by_round"][max(a.rounds)])
summary["chosen"] = best
json.dump(summary, open(out / f"{a.model}.json", "w"), indent=1)
print(json.dumps({k: v["acc_by_round"] for k, v in summary["formats"].items()}), "chosen:", best)
