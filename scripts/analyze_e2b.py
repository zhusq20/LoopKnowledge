"""Summaries for 2B / 2C runs: python scripts/analyze_e2b.py ouro-1.4b <tag> [<tag> ...]

Full set and premise subset. Premise (plan 4.4): after the edit every edited single hop
gives the new answer at the final round, and before the edit the model answered the
original chain (both hops and the nested two-hop) at the final round. For hop1 edits
the model must also know the new bridge's second hop before the edit (new_hop_1).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
model, tags = sys.argv[1], sys.argv[2:]
rng = np.random.default_rng(0)


def ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [float("nan")] * 3
    m = x[rng.integers(0, len(x), (1000, len(x)))].mean(1)
    return [float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def f3(v):
    return f"{v[0]:.2f}" if v[0] == v[0] else "—"


for tag in tags:
    d = ROOT / "results" / "e2b" / model / tag
    args = json.load(open(d / "args.json"))
    recs = [json.loads(l) for l in open(d / "cases.jsonl")]
    T = str(args.get("T") or (4 if model.startswith("ouro") else 32))
    rounds = sorted(recs[0]["after"]["nested"].keys(), key=int)

    def edits_ok(rec, rr=T):
        return all(v[rr]["new"] for k, v in rec["after"].items() if k.startswith("edit_"))

    def premise(rec):
        b = rec["before"]
        ok = b["old_hop_0"][T]["old"] and b["old_hop_1"][T]["old"] and b["nested"][T]["old"] and edits_ok(rec)
        if rec["group"] == "hop1":
            ok = ok and b["new_hop_1"][T]["new"]
        return ok

    S = {"tag": tag, "args": args, "groups": {}}
    L = [f"## {model} / {tag}", "", f"layers={args['layers']} lr={args['lr']} linf={args['linf']} sup={args['sup']} split={args['split']}", ""]
    for g in ["hop1", "hop2", "both"]:
        R = [r for r in recs if r["group"] == g]
        if not R:
            continue
        P = [r for r in R if premise(r)]
        G = {"n": len(R), "n_premise": len(P), "steps": float(np.mean([r["edit_info"]["steps"] for r in R]))}
        for name, sub in [("all", R), ("premise", P)]:
            G[name] = {
                "edit_new": {rr: ci([edits_ok(r, rr) for r in sub]) for rr in rounds},
                "nested_new": {rr: ci([r["after"]["nested"][rr]["new"] for r in sub]) for rr in rounds},
                "nested_old": {rr: ci([r["after"]["nested"][rr]["old"] for r in sub]) for rr in rounds},
                "nested_old_before": {rr: ci([r["before"]["nested"][rr]["old"] for r in sub]) for rr in rounds},
                "new_hop1_new_after": {rr: ci([r["after"]["new_hop_1"][rr]["new"] for r in sub]) for rr in rounds},
                "relnb_new": {rr: ci([any(v[rr]["new"] for k, v in r["after"].items() if k.startswith("relnb_")) for r in sub if any(k.startswith("relnb_") for k in r["after"])]) for rr in rounds},
                "nested_x_new": {rr: ci([r["after"]["nested_x"][rr]["new"] for r in sub if "nested_x" in r["after"]]) for rr in rounds},
                "nested_specific": {rr: ci([r["after"]["nested"][rr]["new"] and not r["after"]["nested_x"][rr]["new"] for r in sub if "nested_x" in r["after"]]) for rr in rounds},
                "para_new": {rr: ci([all(v[rr]["new"] for k, v in r["after"].items() if k.startswith("para_")) for r in sub if "para_0" in r["after"]]) for rr in rounds},
                "nested_q_new": {rr: ci([r["after"]["nested_q"][rr]["new"] for r in sub if "nested_q" in r["after"]]) for rr in rounds},
                "neigh_after": ci([r["neigh_after"][T] for r in sub]),
            }
            if sub and "unroll" in sub[0]:
                G[name]["unroll"] = {
                    k: {
                        "edit_new": ci([all(v[T]["new"] for kk, v in r["unroll"][k].items() if kk.startswith("edit_")) for r in sub]),
                        "nested_new": ci([r["unroll"][k]["nested"][T]["new"] for r in sub]),
                        "nested_old": ci([r["unroll"][k]["nested"][T]["old"] for r in sub]),
                        "neigh": ci([r["unroll_neigh"][k][T] for r in sub]),
                        "relnb_new": ci([any(v[T]["new"] for kk, v in r["unroll"][k].items() if kk.startswith("relnb_")) for r in sub if any(kk.startswith("relnb_") for kk in r["unroll"][k])]),
                        "nested_x_new": ci([r["unroll"][k]["nested_x"][T]["new"] for r in sub if "nested_x" in r["unroll"][k]]),
                        "nested_specific": ci([r["unroll"][k]["nested"][T]["new"] and not r["unroll"][k]["nested_x"][T]["new"] for r in sub if "nested_x" in r["unroll"][k]]),
                    }
                    for k in sub[0]["unroll"]
                }
        S["groups"][g] = G
        L += [f"### {g}: n={len(R)}，前提成立 {len(P)}（{len(P) / max(1, len(R)):.0%}），平均步数 {G['steps']:.1f}，邻居保持 {f3(G['all']['neigh_after'])}", ""]
        L += ["| 退出轮 | 单跳新(全) | 换问法新 | 同关系泄漏 | 两跳新(全) | 换主语两跳新(泄漏) | 特异传播 | 两跳问句新 | 两跳旧(全) | 编辑前两跳旧 | 单跳新(前提) | 两跳新(前提) | 特异传播(前提) |", "|" + "---|" * 13]
        for rr in rounds:
            A, B = G["all"], G["premise"]
            L.append(f"| {rr} | {f3(A['edit_new'][rr])} | {f3(A['para_new'][rr])} | {f3(A['relnb_new'][rr])} | {f3(A['nested_new'][rr])} | {f3(A['nested_x_new'][rr])} | {f3(A['nested_specific'][rr])} | {f3(A['nested_q_new'][rr])} | {f3(A['nested_old'][rr])} | {f3(A['nested_old_before'][rr])} | {f3(B['edit_new'][rr])} | {f3(B['nested_new'][rr])} | {f3(B['nested_specific'][rr])} |")
        if "unroll" in G["all"]:
            L += ["", f"按轮展开（读第 {T} 轮）：", "", "| 施加的轮 | 单跳新(全) | 同关系泄漏 | 两跳新(全) | 换主语两跳新 | 特异传播 | 两跳旧(全) | 邻居 | 单跳新(前提) | 两跳新(前提) |", "|" + "---|" * 10]
            for k, v in G["all"]["unroll"].items():
                pv = G["premise"].get("unroll", {}).get(k, {"edit_new": [float('nan')], "nested_new": [float('nan')]})
                L.append(f"| {k} | {f3(v['edit_new'])} | {f3(v['relnb_new'])} | {f3(v['nested_new'])} | {f3(v['nested_x_new'])} | {f3(v['nested_specific'])} | {f3(v['nested_old'])} | {f3(v['neigh'])} | {f3(pv['edit_new'])} | {f3(pv['nested_new'])} |")
        L.append("")
    json.dump(S, open(d / "summary.json", "w"), indent=1)
    open(d / "summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
