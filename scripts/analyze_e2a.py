"""Summaries for 2A (per-round composition) and 3C (per-round single-hop recall).

python scripts/analyze_e2a.py ouro-1.4b [--final 4]
Writes results/e2a/<model>/summary.json and summary.md. Unit = chain / fact; 95% CIs by
bootstrap over units (1000 resamples). Main split = test (dev was used for the format).
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--final", type=int, default=None, help="round that defines 'known' (default: trained depth)")
p.add_argument("--split", default="test")
a = p.parse_args()

d = json.load(open(ROOT / "results" / "e2a" / a.model / "raw.json"))
R = d["rounds"]
final = a.final or (4 if a.model.startswith("ouro") else 32)
rng = np.random.default_rng(0)


def ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [float("nan")] * 3
    bs = rng.integers(0, len(x), (1000, len(x)))
    m = x[bs].mean(1)
    return [float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


F = {f["key"]: f for f in d["facts"]}
facts = [f for f in d["facts"] if f["split"] == a.split]
chains = [c for c in d["chains"] if c["split"] == a.split]
S = {"model": a.model, "split": a.split, "final_round": final, "rounds": R, "n_facts": len(facts), "n_chains": len(chains)}

# 3C: closed-book single-hop recall by round
S["single_hop_acc"] = {r: ci([f[f"r{r}"]["ok"] for f in facts]) for r in R}

# 2A: chains
def hop_ok(c, r, h):
    return F[c[h]][f"r{r}"]["ok"]

rows = {}
for r in R:
    h1 = np.array([hop_ok(c, r, "hop1") for c in chains])
    h2 = np.array([hop_ok(c, r, "hop2") for c in chains])
    ne = np.array([c[f"r{r}"]["nested_ok"] for c in chains])
    cx = np.array([c[f"r{r}"]["context_ok"] for c in chains])
    co = np.array([c[f"r{r}"].get("corrupt_ok", False) for c in chains])
    both = h1 & h2
    rows[r] = {
        "hop1": ci(h1),
        "hop2": ci(h2),
        "both": ci(both),
        "nested": ci(ne),
        "nested_given_both_same_round": ci(ne[both]),
        "n_both_same_round": int(both.sum()),
        "context_given_both_same_round": ci(cx[both]),
        "context": ci(cx),
        "corrupt_orig_answer": ci(co),
        "nested_given_both_and_not_corrupt_same_round": ci(ne[both & ~co]),
    }
S["chains_by_round"] = rows

# fixed "known two-hop" set: both hops right at the final round, shortcut baseline wrong at final round
known = [c for c in chains if hop_ok(c, final, "hop1") and hop_ok(c, final, "hop2") and not c[f"r{final}"].get("corrupt_ok", False)]
shortcut = [c for c in chains if hop_ok(c, final, "hop1") and hop_ok(c, final, "hop2") and c[f"r{final}"].get("corrupt_ok", False)]
S["known_two_hop"] = {
    "n": len(known),
    "coverage_of_chains": len(known) / max(1, len(chains)),
    "n_shortcut_excluded": len(shortcut),
    "nested_acc_by_round": {r: ci([c[f"r{r}"]["nested_ok"] for c in known]) for r in R},
    "context_acc_by_round": {r: ci([c[f"r{r}"]["context_ok"] for c in known]) for r in R},
    "hop1_acc_by_round": {r: ci([hop_ok(c, r, "hop1") for c in known]) for r in R},
    "hop2_acc_by_round": {r: ci([hop_ok(c, r, "hop2") for c in known]) for r in R},
    "shortcut_nested_acc_by_round": {r: ci([c[f"r{r}"]["nested_ok"] for c in shortcut]) for r in R},
}
# known single-hop facts (right at final round): recall by round = when is a known fact first available
kf = [f for f in facts if f[f"r{final}"]["ok"]]
S["known_single_hop"] = {"n": len(kf), "acc_by_round": {r: ci([f[f"r{r}"]["ok"] for f in kf]) for r in R}}

out = ROOT / "results" / "e2a" / a.model
json.dump(S, open(out / "summary.json", "w"), indent=1)

f3 = lambda v: f"{v[0]:.3f}"
L = [f"# 2A / 3C — {a.model}（{a.split}，已知 = 第 {final} 轮答对）", ""]
L += [f"事实 {len(facts)} 条，两跳链 {len(chains)} 条。已知单跳 {len(kf)} 条。已知两跳 {len(known)} 条（另有 {len(shortcut)} 条被捷径基线排除）。", ""]
L += ["| 轮 | 单跳(全部) | 已知单跳 | hop1 | hop2 | 两跳嵌套 | 两跳｜两单跳同轮对 | 上下文｜两单跳同轮对 | 捷径基线 | 已知两跳：嵌套 | 已知两跳：上下文 |", "|" + "---|" * 11]
for r in R:
    x = rows[r]
    L.append(
        f"| {r} | {f3(S['single_hop_acc'][r])} | {f3(S['known_single_hop']['acc_by_round'][r])} | {f3(x['hop1'])} | {f3(x['hop2'])} | {f3(x['nested'])} | "
        f"{f3(x['nested_given_both_same_round'])} (n={x['n_both_same_round']}) | {f3(x['context_given_both_same_round'])} | {f3(x['corrupt_orig_answer'])} | "
        f"{f3(S['known_two_hop']['nested_acc_by_round'][r])} | {f3(S['known_two_hop']['context_acc_by_round'][r])} |"
    )
open(out / "summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
