"""2B (propagation of a single-point edit, by exit round and hop position) and
2C (which round's copy of the edit carries it: apply the same dW only in chosen rounds).

Two-hop MQuAKE-CF-3k-v2 cases, one case at a time, weights restored after each case.
Groups: hop1 (edit_triples_idx == [0]), hop2 ([1]), both ([0, 1]).
Per case and exit round we record new / old answer matches for:
  edit_k   : each edited single-hop cloze (efficacy)
  new_hop_k: the new chain's single hops (premise: does the model know the new chain?)
  nested   : the two-hop nested cloze (propagation)
and, at the final round, retention of fixed unrelated known facts (neighbours).
Pre-edit answers for the same prompts are stored too (premise: the base model knows the
original chain). With --unroll, the same dW is also applied only in the listed round
subsets (2C) and the same prompts are re-read.

python scripts/e2b_edit.py --model ouro-1.4b --layers 4 5 6 --split dev --n 40 --tag dev-l456
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch

from lk.data import alias_list, facts_and_chains, load_cases, match, nested, split_of, template_of
from lk.edit import ft_edit
from lk.evalrounds import FEWSHOT, FEWSHOT_QA, eval_prompts, qa
from lk.loopmodel import Edit, load

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--layers", nargs="+", required=True, help="int layer ids (Huginn: prelude:i / coda:i allowed)")
p.add_argument("--T", type=int, default=None)
p.add_argument("--read-rounds", type=int, nargs="+", default=None)
p.add_argument("--split", default="dev")
p.add_argument("--groups", nargs="+", default=["hop1", "hop2", "both"])
p.add_argument("--n", type=int, default=40, help="cases per group")
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--linf", type=float, default=None)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--stop-p", type=float, default=0.9)
p.add_argument("--sup", default="final", choices=["final", "all"])
p.add_argument("--unroll", default="", help="2C subsets, e.g. '1;2;3;4;1,2;2,3,4;3,4'")
p.add_argument("--n-neighbors", type=int, default=24)
p.add_argument("--tag", required=True)
a = p.parse_args()

m = load(a.model)
T = a.T or (4 if m.kind == "ouro" else 32)
RR = a.read_rounds or list(range(1, T + 1))
layers = [(x.split(":")[0], int(x.split(":")[1])) if ":" in x else int(x) for x in a.layers]
unroll = [set(int(r) for r in s.split(",")) for s in a.unroll.split(";")] if a.unroll else []
out = ROOT / "results" / "e2b" / a.model / a.tag
out.mkdir(parents=True, exist_ok=True)
log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
json.dump(vars(a), open(out / "args.json", "w"), indent=1)

cases = [c for c in load_cases() if len(c["single_hops"]) == 2 and split_of(c["case_id"]) == a.split]
gmap = {(0,): "hop1", (1,): "hop2", (0, 1): "both"}
by_group = {}
for c in cases:
    by_group.setdefault(gmap[tuple(c["orig"]["edit_triples_idx"])], []).append(c)
rng = random.Random(0)
sel = []
for g in a.groups:
    lst = list(by_group[g])
    rng.shuffle(lst)
    sel += [(g, c) for c in lst[: a.n]]
log(f"{len(sel)} cases: " + ", ".join(f"{g}={sum(1 for x in sel if x[0] == g)}" for g in a.groups))

# neighbours: known unrelated facts (correct at T), from the test split of facts, disjoint from selected cases' subjects
facts, chains_all = facts_and_chains(load_cases())
by_rel = {}
for f in facts.values():
    by_rel.setdefault(f["rel"], []).append(f)
chains_by_r1 = {}
for ch in chains_all.values():
    chains_by_r1.setdefault(ch["r1_label"], []).append(ch)
subj_used = {h["cloze"] for _, c in sel for h in c["single_hops"] + c["new_single_hops"]}
pool = [f for f in facts.values() if f["split"] == "test" and f["cloze"] not in subj_used]
random.Random(2).shuffle(pool)
pool = pool[: 6 * a.n_neighbors]
pre = eval_prompts(m, [f["cloze"] for f in pool], {"ok": [f["aliases"] for f in pool]}, [T], FEWSHOT, batch=64)
neigh = [f for f, ok in zip(pool, pre[T]["ok"]) if ok][: a.n_neighbors]
log(f"{len(neigh)} neighbour facts")


def case_prompts(c):
    """list of (name, text, new_aliases, old_aliases)."""
    P = []
    for k, rw in enumerate(c["requested_rewrite"]):
        cl = rw["prompt"].format(rw["subject"])
        P.append((f"edit_{k}", cl, alias_list(rw["target_new"]["str"], []), alias_list(rw["target_true"]["str"], [])))
    for k, (h, nh) in enumerate(zip(c["single_hops"], c["new_single_hops"])):
        P.append((f"old_hop_{k}", h["cloze"], alias_list(nh["answer"], nh["answer_alias"]), alias_list(h["answer"], h["answer_alias"])))
        if nh["cloze"] != h["cloze"]:
            P.append((f"new_hop_{k}", nh["cloze"], alias_list(nh["answer"], nh["answer_alias"]), alias_list(h["answer"], h["answer_alias"]) if k == 0 else []))
        else:
            P.append((f"new_hop_{k}", nh["cloze"], alias_list(nh["answer"], nh["answer_alias"]), alias_list(h["answer"], h["answer_alias"])))
    tl = c["orig"]["triples_labeled"]
    t2 = template_of(c["single_hops"][1]["cloze"], tl[1][0])
    P.append(("nested", nested(t2, tl[0][1], tl[0][0]), alias_list(c["new_answer"], c["new_answer_alias"]), alias_list(c["answer"], c["answer_alias"])))
    # leakage controls. relnb_k: same relation as edit k, different subject (does the edit leak to the relation?)
    crng = random.Random(c["case_id"])
    for k, rw in enumerate(c["requested_rewrite"]):
        newa = set(alias_list(rw["target_new"]["str"], []))
        cand = [f for f in by_rel.get(rw["relation_id"], []) if f["subject"] != rw["subject"] and not (set(f["aliases"]) & newa) and f["subject"] not in rw["target_new"]["str"]]
        if cand:
            f = crng.choice(cand)
            P.append((f"relnb_{k}", f["cloze"], list(newa), f["aliases"]))
    # nested_x: two-hop prompt with s1 replaced by another subject (same r1, different bridge, different answer)
    olda, newa2 = set(alias_list(c["answer"], c["answer_alias"])), set(alias_list(c["new_answer"], c["new_answer_alias"]))
    cand = [ch for ch in chains_by_r1.get(tl[0][1], []) if ch["s1"] != tl[0][0] and ch["bridge"] not in (tl[0][2], c["new_single_hops"][0]["answer"]) and not (set(ch["aliases"]) & (olda | newa2))]
    if cand:
        ch = crng.choice(cand)
        P.append(("nested_x", nested(t2, tl[0][1], ch["s1"]), list(newa2), list(olda)))
    # question forms (few-shot QA): paraphrase of each edited fact, and the original two-hop question
    for k, rw in enumerate(c["requested_rewrite"]):
        P.append((f"qa:para_{k}", qa(rw["question"]), alias_list(rw["target_new"]["str"], []), alias_list(rw["target_true"]["str"], [])))
    P.append(("qa:nested_q", qa(c["questions"][0]), alias_list(c["new_answer"], c["new_answer_alias"]), alias_list(c["answer"], c["answer_alias"])))
    return P


def read(prompts, rounds, edit):
    out = {}
    for pre, sel in [(FEWSHOT, [x for x in prompts if not x[0].startswith("qa:")]), (FEWSHOT_QA, [x for x in prompts if x[0].startswith("qa:")])]:
        if not sel:
            continue
        r = eval_prompts(m, [x[1] for x in sel], {"new": [x[2] for x in sel], "old": [x[3] for x in sel]}, rounds, pre, batch=64, edit=edit)
        for i, x in enumerate(sel):
            out[x[0].replace("qa:", "")] = {str(rr): {"gen": r[rr]["gen"][i], "new": r[rr]["new"][i], "old": r[rr]["old"][i]} for rr in rounds}
    return out


def read_neigh(rounds, edit):
    r = eval_prompts(m, [f["cloze"] for f in neigh], {"ok": [f["aliases"] for f in neigh]}, rounds, FEWSHOT, batch=64, edit=edit)
    return {str(rr): sum(r[rr]["ok"]) / max(1, len(neigh)) for rr in rounds}


sup = [T] if a.sup == "final" else list(range(1, T + 1))
fout = open(out / "cases.jsonl", "w")
t0 = time.time()
for i, (g, c) in enumerate(sel):
    P = case_prompts(c)
    before = read(P, RR, None)
    items = [(rw["prompt"].format(rw["subject"]), rw["target_new"]["str"]) for rw in c["requested_rewrite"]]
    edit, info = ft_edit(m, items, layers, T, prefixes=(FEWSHOT, ""), steps=a.steps, lr=a.lr, linf=a.linf, sup_rounds=sup, stop_p=a.stop_p)
    after = read(P, RR, edit)
    rec = {"case_id": c["case_id"], "group": g, "edit_info": info, "before": before, "after": after, "neigh_after": read_neigh([T], edit)}
    for sub in unroll:
        e2 = Edit(edit.deltas, sub)
        key = ",".join(map(str, sorted(sub)))
        rec.setdefault("unroll", {})[key] = read(P, [T], e2)
        rec.setdefault("unroll_neigh", {})[key] = read_neigh([T], e2)
    fout.write(json.dumps(rec) + "\n")
    fout.flush()
    if i % 10 == 0:
        ar = rec["after"]
        log(f"[{i}/{len(sel)}] {g} steps={info['steps']} edit_new@T={ar['edit_0'][str(T)]['new']} nested new/old@T={ar['nested'][str(T)]['new']}/{ar['nested'][str(T)]['old']} neigh={rec['neigh_after'][str(T)]:.2f} {time.time() - t0:.0f}s")
log("done")
