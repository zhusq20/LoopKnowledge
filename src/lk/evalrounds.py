"""Greedy 10-token answers at every exit round, alias match against one or more answer sets."""

from __future__ import annotations

import time

import torch

from .data import match
from .loopmodel import greedy_at_round, greedy_cached

FEWSHOT = (
    "The capital of Japan is Tokyo.\n"
    "The Eiffel Tower is located in the city of Paris.\n"
    "Barack Obama is married to Michelle Obama.\n"
    "The author of Hamlet is William Shakespeare.\n"
)
FEWSHOT_QA = (
    "Q: What is the capital of Japan?\nA: Tokyo\n"
    "Q: In which city is the Eiffel Tower located?\nA: Paris\n"
    "Q: Who is Barack Obama married to?\nA: Michelle Obama\n"
    "Q: Who is the author of Hamlet?\nA: William Shakespeare\n"
)
FORMATS = {"zeroshot": "", "fewshot": FEWSHOT}


def qa(question):
    return f"Q: {question}\nA:"


def encode(model, text, prefix=""):
    ids = model.tok(prefix + text, add_special_tokens=False).input_ids
    if model.bos is not None:
        ids = [model.bos] + ids
    return ids


def cut(model, toks):
    s = model.tok.decode(toks, skip_special_tokens=True)
    return s.split("\n")[0]


@torch.no_grad()
def eval_prompts(model, texts, alias_sets, rounds, prefix="", max_new=10, batch=128, hooks=None, edit=None, log=None):
    """alias_sets: dict name -> list (per prompt) of alias lists.
    Returns dict round -> {"gen": [str], name: [bool]}."""
    enc = [encode(model, t, prefix) for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out = {}
    for r in rounds:
        t0 = time.time()
        gen = [None] * len(enc)
        for b in range(0, len(order), batch):
            idx = order[b : b + batch]
            if hooks is None:
                g = greedy_cached(model, [enc[i] for i in idx], r, max_new, edit)
            else:
                g = greedy_at_round(model, [enc[i] for i in idx], r, max_new, hooks, edit)
            for i, toks in zip(idx, g):
                gen[i] = cut(model, toks)
        res = {"gen": gen}
        for name, al in alias_sets.items():
            res[name] = [match(g, a) for g, a in zip(gen, al)]
        out[r] = res
        if log:
            log(f"round {r}: {time.time() - t0:.1f}s " + " ".join(f"{k}={sum(v) / len(v):.3f}" for k, v in res.items() if k != "gen"))
    return out
