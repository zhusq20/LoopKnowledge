"""Constrained fine-tuning edits on MLP down_proj deltas of chosen layers (shared weights).

The delta dW is added in every round (ordinary shared-weight edit). The loss is the NLL
of the new answer tokens read out at the supervised rounds (default: final round only).
||dW||_inf is clamped to `linf` after every step (FT-L style constraint). Early stop once
every edit prompt gives the new answer's tokens mean prob >= stop_p at the final round.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .loopmodel import Edit, pad_right


def seq_logits(model, seqs, T, edit=None, rounds=None, hooks=None):
    """Per-round logits at all positions -> dict r -> [B, L, V] (fp32) for r in rounds."""
    ids, _ = pad_right(seqs, model.pad_id, model.device)
    out = {}
    for r, x in model.rounds(ids, T, hooks, edit):
        if rounds is None or r in rounds:
            if model.kind == "huginn":
                out[r] = model.readout(x, None, r=r, hooks=hooks, edit=edit)
            else:
                out[r] = model.readout(x)
    return out


def build_examples(model, items, prefixes):
    """items: list of (prompt_text, target_text). Returns list of (ids, target_positions, target_ids)."""
    ex = []
    for prompt, target in items:
        for pre in prefixes:
            p = model.tok(pre + prompt, add_special_tokens=False).input_ids
            if model.bos is not None:
                p = [model.bos] + p
            t = model.tok(" " + target, add_special_tokens=False).input_ids
            ids = p + t
            pos = list(range(len(p) - 1, len(ids) - 1))
            ex.append((ids, pos, t))
    return ex


def nll(model, ex, edit, T, sup_rounds, weights=None):
    seqs = [e[0] for e in ex]
    lg = seq_logits(model, seqs, T, edit, rounds=set(sup_rounds))
    loss, probs = 0.0, None
    for k, r in enumerate(sup_rounds):
        lp = lg[r].log_softmax(-1)
        per = []
        for i, (_, pos, t) in enumerate(ex):
            per.append(lp[i, pos, t].mean())
        per = torch.stack(per)
        w = 1.0 if weights is None else weights[k]
        loss = loss - w * per.mean()
        if r == T:
            probs = per.detach().exp()
    return loss, probs


def ft_edit(model, items, layers, T, prefixes=("",), steps=40, lr=5e-4, linf=None, sup_rounds=None, round_weights=None, stop_p=0.9, wd=0.0, log=None):
    sup_rounds = list(sup_rounds or [T])
    if T not in sup_rounds:
        raise ValueError("final round must be supervised (used for early stop)")
    deltas = {l: torch.zeros_like(model.down_proj(l), dtype=torch.float32, requires_grad=True) for l in layers}
    edit = Edit(deltas, None)
    opt = torch.optim.Adam(deltas.values(), lr=lr, weight_decay=wd)
    ex = build_examples(model, items, prefixes)
    hist = []
    with torch.enable_grad():
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            loss, probs = nll(model, ex, edit, T, sup_rounds, round_weights)
            hist.append((float(loss), float(probs.min())))
            if probs.min() >= stop_p:
                break
            loss.backward()
            opt.step()
            if linf is not None:
                with torch.no_grad():
                    for d in deltas.values():
                        d.clamp_(-linf, linf)
    out = Edit({l: d.detach() for l, d in deltas.items()}, None)
    info = {"steps": len(hist), "final_loss": hist[-1][0], "final_min_p": hist[-1][1], "norms": {str(l): float(d.norm()) for l, d in out.deltas.items()}}
    if log:
        log(info)
    return out, info
