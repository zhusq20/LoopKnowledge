"""1A (zero training): where is a known fact recalled, by unrolled (round, layer) site.

ROME causal tracing on the unrolled depth axis. Subject-token embeddings get Gaussian
noise (3 x std of subject-token embeddings); the clean value is restored at one site
and the probability of the clean answer token at the final round is read.
Maps over unrolled depth d (execution order of all layer calls; Huginn adds prelude
and coda sites):
  hid_subj / hid_last   : restore the residual stream after site d at subject-last / last pos
  mlp_subj / attn_subj / mlp_last / attn_last : restore that branch over a centred
                          window of W unrolled sites
Knockouts on the clean run (Geva et al. 2023):
  ko_attn  : last position may not attend to subject positions in a window of W sites
  ko_mlp   : MLP output zeroed at all subject positions in a window of W sites
Measurement only, nothing is fitted.

python scripts/e1a_trace.py --model ouro-1.4b --T 4 --n 200
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

from lk.data import facts_and_chains, load_cases
from lk.evalrounds import FEWSHOT, eval_prompts
from lk.loopmodel import Hooks, load, round_logits_last

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--T", type=int, required=True, help="rounds; the answer is read at round T")
p.add_argument("--n", type=int, default=200)
p.add_argument("--noise-samples", type=int, default=5)
p.add_argument("--window", type=int, default=5)
p.add_argument("--chunk", type=int, default=160)
p.add_argument("--split", default="test")
a = p.parse_args()

torch.set_grad_enabled(False)
out = ROOT / "results" / "e1a" / a.model
out.mkdir(parents=True, exist_ok=True)
m = load(a.model)
log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)

# ---------------------------------------------------------------- sites
if m.kind == "ouro":
    sites = [(r, l) for r in range(1, a.T + 1) for l in range(m.n_layers)]
else:
    sites = [(0, ("prelude", i)) for i in range(len(m.tr.prelude))]
    sites += [(r, l) for r in range(1, a.T + 1) for l in range(m.n_layers)]
    sites += [(a.T, ("coda", i)) for i in range(len(m.tr.coda))]
D = len(sites)
sidx = {s: i for i, s in enumerate(sites)}
half = a.window // 2
win = [list(range(max(0, d - half), min(D, d + half + 1))) for d in range(D)]

# ---------------------------------------------------------------- items: known facts
facts, _ = facts_and_chains(load_cases())
cand = [f for f in facts.values() if f["split"] == a.split]
random.Random(1).shuffle(cand)
cand = cand[: 4 * a.n]
res = eval_prompts(m, [f["cloze"] for f in cand], {"ok": [f["aliases"] for f in cand]}, [a.T], FEWSHOT, batch=64)
known = [f for f, ok in zip(cand, res[a.T]["ok"]) if ok]
log(f"{len(known)} known facts of {len(cand)} candidates")


def encode_with_subject(f):
    text = FEWSHOT + f["cloze"]
    enc = m.tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc.input_ids, enc.offset_mapping
    c0 = len(FEWSHOT) + f["cloze"].index(f["subject"])
    c1 = c0 + len(f["subject"])
    subj = [i for i, (s, e) in enumerate(offs) if s < c1 and e > c0]
    if m.bos is not None:
        ids = [m.bos] + ids
        subj = [i + 1 for i in subj]
    return ids, subj


# noise level: 3 x std of (scaled) subject-token embeddings, as ROME
E = m.embed_matrix()
scale = getattr(m.lm, "emb_scale", 1.0) if m.kind == "huginn" else 1.0
sub_tok = set()
for f in facts.values():
    sub_tok.update(m.tok(" " + f["subject"], add_special_tokens=False).input_ids)
noise_level = 3 * (E[list(sub_tok)].float() * scale).std().item()
log(f"noise level {noise_level:.4f}")


def run(ids, B, hooks):
    return round_logits_last(m, [ids] * B, a.T, hooks, rounds_keep={a.T})[0].softmax(-1)


results, skipped = [], []
t0 = time.time()
for it, f in enumerate(known):
    ids, subj = encode_with_subject(f)
    S = len(ids)
    sl, last = subj[-1], S - 1
    pos = {"subj": sl, "last": last}
    rec = {"hid": {}, "mlp": {}, "attn": {}}

    def rec_hook(kind):
        def h(r, li, x):
            rec[kind][(r, li)] = x[0, [sl, last]].clone()
            return x
        return h

    probs = run(ids, 1, Hooks(after_layer=rec_hook("hid"), mlp_out=rec_hook("mlp"), attn_out=rec_hook("attn")))[0]
    target = int(probs.argmax())
    p_clean = float(probs[target])
    if m.tok.decode([target]).strip().lower() in {"", "the", "a", "an"}:
        skipped.append(f["key"])
        continue
    if len(results) >= a.n:
        break

    # rows: 0 = corrupted baseline, then 6 maps x D restorations
    maps = ["hid_subj", "hid_last", "mlp_subj", "attn_subj", "mlp_last", "attn_last"]
    rows = [("none", -1)] + [(mp, d) for mp in maps for d in range(D)]
    acc = np.zeros(len(rows))
    for ns in range(a.noise_samples):
        g = torch.Generator(device=m.device).manual_seed(1000 * it + ns)
        noise = torch.randn(len(subj), E.shape[1], generator=g, device=m.device, dtype=torch.float32)
        for c0 in range(0, len(rows), a.chunk):
            chunk = rows[c0 : c0 + a.chunk]
            B = len(chunk)
            # which rows restore what at which site
            hid_at, br_at = {}, {"mlp": {}, "attn": {}}
            for j, (mp, d) in enumerate(chunk):
                if mp == "none":
                    continue
                kind, where = mp.split("_")
                if kind == "hid":
                    hid_at.setdefault(sites[d], []).append((j, where))
                else:
                    for dd in win[d]:
                        br_at[kind].setdefault(sites[dd], []).append((j, where))

            def emb_hook(x):
                x = x.clone()
                x[:, subj] = (x[:, subj].float() + noise_level * noise).to(x.dtype)
                return x

            def make(kind, table):
                def h(r, li, x):
                    lst = table.get((r, li))
                    if not lst:
                        return x
                    x = x.clone()
                    for j, where in lst:
                        k = 0 if where == "subj" else 1
                        x[j, pos[where]] = rec[kind][(r, li)][k]
                    return x
                return h

            hk = Hooks(embed=emb_hook, after_layer=make("hid", hid_at), mlp_out=make("mlp", br_at["mlp"]), attn_out=make("attn", br_at["attn"]))
            pr = run(ids, B, hk)[:, target].float().cpu().numpy()
            acc[c0 : c0 + B] += pr
    acc /= a.noise_samples
    p_corrupt = float(acc[0])
    item = {"key": f["key"], "cloze": f["cloze"], "target": m.tok.decode([target]), "p_clean": p_clean, "p_corrupt": p_corrupt, "n_subj_tok": len(subj)}
    for mi, mp in enumerate(maps):
        item[mp] = acc[1 + mi * D : 1 + (mi + 1) * D].tolist()

    # knockouts on the clean run
    ko_rows = [("ko_attn", d) for d in range(D)] + [("ko_mlp", d) for d in range(D)]
    ko = np.zeros(len(ko_rows))
    for c0 in range(0, len(ko_rows), a.chunk):
        chunk = ko_rows[c0 : c0 + a.chunk]
        B = len(chunk)
        att_rows, mlp_rows = {}, {}
        for j, (kind, d) in enumerate(chunk):
            for dd in win[d]:
                (att_rows if kind == "ko_attn" else mlp_rows).setdefault(sites[dd], []).append(j)

        def mask_hook(r, li):
            lst = att_rows.get((r, li))
            if not lst:
                return None
            mk = torch.ones(B, 1, S, S, dtype=torch.bool, device=m.device)
            for j in lst:
                mk[j, 0, last, subj] = False
            return mk

        def zero_hook(r, li, y):
            lst = mlp_rows.get((r, li))
            if not lst:
                return y
            y = y.clone()
            for j in lst:
                y[j, subj] = 0
            return y

        pr = run(ids, B, Hooks(attn_mask=mask_hook, mlp_out=zero_hook))[:, target].float().cpu().numpy()
        ko[c0 : c0 + B] = pr
    item["ko_attn"] = ko[:D].tolist()
    item["ko_mlp"] = ko[D:].tolist()
    results.append(item)
    if it % 10 == 0:
        log(f"item {it}/{len(known)} p_clean={p_clean:.3f} p_corrupt={p_corrupt:.3f} elapsed {time.time() - t0:.0f}s")

json.dump({"model": a.model, "T": a.T, "sites": [[r, li] for r, li in sites], "window": a.window, "noise_level": noise_level, "noise_samples": a.noise_samples, "skipped_article_target": skipped, "items": results}, open(out / f"trace_T{a.T}.json", "w"))
log("done")
