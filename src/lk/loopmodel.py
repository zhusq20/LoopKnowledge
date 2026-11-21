"""Round-by-round forward of looped LMs with hooks, per-round weight deltas and readout.

Batches are right-padded. Attention is plain causal, so a real position never sees a
pad position, and every row is read at its own last real position. No KV cache: each
call recomputes the full sequence. Prompts here are short, so this is cheap and it keeps
"exit at round r" exact (the r-round model computes every position with r rounds).

Ouro: x_0 = embed(ids). Round r runs all layers, then model.norm. Readout of round r is
lm_head(x_r), identical to the official hidden_states_list[r-1].

Huginn: e = prelude(scale * wte(ids)). x_0 is a fixed random state (seeded table, as in
llm-memory-editability/scripts/loop_depth_dynamics.py). Round r: x = core(adapter([x, e])).
Readout of round r runs the coda on ln_f(x_r) over all positions, then ln_f and lm_head.

Hooks (all optional) are passed in a `Hooks` object:
  - after_layer(r, l, x) -> x : residual stream after layer l of round r (r is 1-based).
    For Huginn l indexes core blocks; prelude/coda use r=0 / r=-1.
  - mlp_out(r, l, y) -> y     : MLP branch output (before the post-norm for Ouro and
    after norm_4 input sum for Huginn this is the raw MLP output).
  - attn_mask(r, l) -> bool mask [B,1,S,S] or None : extra mask (True = may attend),
    combined with causality. Used for attention knockout.
  - embed(x) -> x             : token embeddings (Ouro) / scaled wte before prelude (Huginn).
Per-round weight deltas: `deltas = {layer: dW}` with dW shaped like down_proj.weight
([hidden, intermediate]) and `delta_rounds` = set of 1-based rounds where they apply.
For Huginn the layer index is the core-block index (0..3) and the prelude/coda blocks
can be addressed as ("prelude", i) / ("coda", i).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

MODEL_ROOT = Path("/mnt/disk2_from_server2/siqizhu4/models")


@dataclass
class Hooks:
    after_layer: Callable | None = None
    mlp_out: Callable | None = None
    attn_mask: Callable | None = None
    embed: Callable | None = None
    attn_out: Callable | None = None  # (r, l, o) -> o : attention branch output before its post-norm


@dataclass
class Edit:
    deltas: dict = field(default_factory=dict)  # layer key -> dW (fp32), same shape as down_proj
    rounds: set | None = None  # None = every round (ordinary shared-weight edit)

    def get(self, key, r):
        if key not in self.deltas:
            return None
        if self.rounds is not None and r not in self.rounds:
            return None
        return self.deltas[key]


class KV:
    """Per-site KV cache for left-padded batched decoding. valid[b, j] = key j is a real token."""

    def __init__(self, valid):
        self.valid = valid
        self.store = {}
        self.start = 0

    def attend(self, key, q, k, v, scale=None):
        s = q.shape[2]
        end = self.start + s
        if key not in self.store:
            B, H, _, D = k.shape
            Lmax = self.valid.shape[1]
            self.store[key] = (k.new_zeros(B, H, Lmax, D), v.new_zeros(B, H, Lmax, D))
        K, V = self.store[key]
        K[:, :, self.start : end] = k
        V[:, :, self.start : end] = v
        dev = q.device
        qi = torch.arange(self.start, end, device=dev)
        ki = torch.arange(end, device=dev)
        m = (ki[None] <= qi[:, None])[None, None] & self.valid[:, None, None, :end]
        m = m | (ki[None] == qi[:, None])[None, None]
        return F.scaled_dot_product_attention(q, K[:, :, :end], V[:, :, :end], attn_mask=m, scale=scale)


def _mask(S, device, extra):
    causal = torch.ones(S, S, dtype=torch.bool, device=device).tril()
    if extra is None:
        return None, True
    return (causal[None, None] & extra), False


class OuroLoop:
    kind = "ouro"

    def __init__(self, name="ouro-1.4b", device="cuda", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = MODEL_ROOT / name
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(path)
        lm = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype, attn_implementation="sdpa"
        )
        self.lm = lm.to(device).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.inner = self.lm.model
        cfg = self.lm.config
        self.cfg = cfg
        self.T_train = cfg.total_ut_steps
        self.n_layers = cfg.num_hidden_layers
        self.layers = list(self.inner.layers[: self.n_layers])
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.rope = sys.modules[type(self.inner).__module__].apply_rotary_pos_emb
        self.device = torch.device(device)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.bos = None  # Ouro base is used without BOS (as in loop_suffix / LoopExperiments)

    # parts used by editing / tracing code
    def mlp_keys(self):
        return list(range(self.n_layers))

    def down_proj(self, key):
        return self.layers[key].mlp.down_proj.weight

    def embed_matrix(self):
        return self.inner.embed_tokens.weight

    def _layer(self, r, li, x, cos, sin, hooks, edit, kv=None):
        layer = self.layers[li]
        b, s, _ = x.shape
        a = layer.self_attn
        h = layer.input_layernorm(x)
        q = a.q_proj(h).view(b, s, -1, self.head_dim).transpose(1, 2)
        k = a.k_proj(h).view(b, s, -1, self.head_dim).transpose(1, 2)
        v = a.v_proj(h).view(b, s, -1, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, cos, sin)
        if kv is not None:
            o = kv.attend((r, li), q, k, v, a.scaling)
        else:
            extra = hooks.attn_mask(r, li) if hooks and hooks.attn_mask else None
            m, causal = _mask(s, x.device, extra)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=causal, scale=a.scaling)
        o = a.o_proj(o.transpose(1, 2).reshape(b, s, -1))
        if hooks and hooks.attn_out:
            o = hooks.attn_out(r, li, o)
        x = x + layer.input_layernorm_2(o)
        mlp = layer.mlp
        u = layer.post_attention_layernorm(x)
        act = mlp.act_fn(mlp.gate_proj(u)) * mlp.up_proj(u)
        y = mlp.down_proj(act)
        dW = edit.get(li, r) if edit else None
        if dW is not None:
            y = y + (act.float() @ dW.T).to(y.dtype)
        if hooks and hooks.mlp_out:
            y = hooks.mlp_out(r, li, y)
        x = x + layer.post_attention_layernorm_2(y)
        if hooks and hooks.after_layer:
            x = hooks.after_layer(r, li, x)
        return x

    def rounds(self, ids, T, hooks=None, edit=None, x0=None, pos=None, kv=None):
        """Yield (r, x_r) for r = 1..T. x_r is normed (as the model feeds it on)."""
        x = self.inner.embed_tokens(ids) if x0 is None else x0
        if hooks and hooks.embed:
            x = hooks.embed(x)
        if pos is None:
            pos = torch.arange(ids.shape[1], device=ids.device)[None].expand(ids.shape[0], -1)
        cos, sin = self.inner.rotary_emb(x, pos)
        for r in range(1, T + 1):
            for li in range(self.n_layers):
                x = self._layer(r, li, x, cos, sin, hooks, edit, kv)
            x = self.inner.norm(x)
            yield r, x

    def readout(self, x, sel=None, ids=None):
        """Logits (fp32). sel: (rows, cols) index tensors or None for all positions."""
        h = x if sel is None else x[sel]
        return self.lm.lm_head(h).float()


class HuginnLoop:
    kind = "huginn"

    def __init__(self, name="huginn-0125", device="cuda", dtype=torch.bfloat16, seed=0, max_len=4096):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = MODEL_ROOT / name
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(path)
        lm = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=dtype)
        self.lm = lm.to(device).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        cfg = self.lm.config
        self.cfg = cfg
        self.T_train = cfg.mean_recurrence
        self.tr = self.lm.transformer
        self.n_layers = len(self.tr.core_block)
        self.device = torch.device(device)
        std = cfg.init_values["std"]
        table = torch.empty(max_len, cfg.n_embd)
        gen = torch.Generator().manual_seed(seed)
        torch.nn.init.trunc_normal_(table, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=gen)
        if self.lm.emb_scale != 1:
            table = table * self.lm.emb_scale
        self.init = table.to(device=device, dtype=dtype)
        self.pad_id = cfg.pad_token_id
        self.bos = cfg.bos_token_id

    def mlp_keys(self):
        return [("prelude", i) for i in range(len(self.tr.prelude))] + list(range(self.n_layers)) + [
            ("coda", i) for i in range(len(self.tr.coda))
        ]

    def block(self, key):
        if isinstance(key, tuple):
            return getattr(self.tr, key[0])[key[1]]
        return self.tr.core_block[key]

    def down_proj(self, key):
        return self.block(key).mlp.proj.weight

    def embed_matrix(self):
        return self.tr.wte.weight

    def _block(self, blk, key, r, li, x, freqs, hooks, edit, kv=None):
        a = blk.attn
        B, S, E = x.shape
        q, k, v = a.Wqkv(blk.norm_1(x)).split(a.chunks, dim=2)
        q = q.view(B, S, a.n_head, a.head_dim)
        k = k.view(B, S, a.n_kv_heads, a.head_dim)
        v = v.view(B, S, a.n_kv_heads, a.head_dim)
        if a.config.qk_bias:
            qb, kb = a.qk_bias.split(1, dim=0)
            q, k = (q + qb).to(q.dtype), (k + kb).to(q.dtype)
        q, k = self._rot(q, k, freqs)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if kv is not None:
            y = kv.attend(key if isinstance(key, tuple) else (r, key), q, k, v)
        else:
            extra = hooks.attn_mask(r, li) if hooks and hooks.attn_mask else None
            m, causal = _mask(S, x.device, extra)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=causal)
        y = a.proj(y.transpose(1, 2).reshape(B, S, E))
        if hooks and hooks.attn_out:
            y = hooks.attn_out(r, li, y)
        x = blk.norm_2(y + x)
        u = blk.norm_3(x)
        f1, f2 = blk.mlp.fc(u).chunk(2, dim=-1)
        act = blk.mlp.nonlin(f1) * f2
        out = blk.mlp.proj(act)
        dW = edit.get(key, r) if edit else None
        if dW is not None:
            out = out + (act.float() @ dW.T).to(out.dtype)
        if hooks and hooks.mlp_out:
            out = hooks.mlp_out(r, li, out)
        x = blk.norm_4(out + x)
        if hooks and hooks.after_layer:
            x = hooks.after_layer(r, li, x)
        return x

    def _rot(self, q, k, freqs):
        mod = sys.modules[type(self.lm).__module__]
        return mod.apply_rotary_emb_complex_like(q, k, freqs_cis=freqs)

    def prelude(self, ids, hooks=None, edit=None, pos=None, kv=None):
        lm = self.lm
        freqs = lm.freqs_cis[:, : ids.shape[1]] if pos is None else lm.freqs_cis[0][pos]
        e = self.tr.wte(ids)
        if lm.emb_scale != 1:
            e = e * lm.emb_scale
        if hooks and hooks.embed:
            e = hooks.embed(e)
        for i, blk in enumerate(self.tr.prelude):
            e = self._block(blk, ("prelude", i), 0, ("prelude", i), e, freqs, hooks, edit, kv)
        return e, freqs

    def rounds(self, ids, T, hooks=None, edit=None, x0=None, pos=None, kv=None):
        e, freqs = self.prelude(ids, hooks, edit, pos, kv)
        self._freqs = freqs
        B, L = ids.shape
        if x0 is not None:
            x = x0
        elif pos is None:
            x = self.init[:L][None].expand(B, -1, -1).clone()
        else:
            x = self.init[pos].clone()
        for r in range(1, T + 1):
            x = self.tr.adapter(torch.cat([x, e], dim=-1))
            for li, blk in enumerate(self.tr.core_block):
                x = self._block(blk, li, r, li, x, freqs, hooks, edit, kv)
            yield r, x

    def readout(self, x, sel=None, ids=None, r=None, hooks=None, edit=None, kv=None):
        freqs = self._freqs
        y = self.tr.ln_f(x)
        for i, blk in enumerate(self.tr.coda):
            y = self._block(blk, ("coda", i), r if r is not None else -1, ("coda", i), y, freqs, hooks, edit, kv)
        y = self.tr.ln_f(y)
        h = y if sel is None else y[sel]
        return self.lm.lm_head(h).float()


class QwenFF(OuroLoop):
    """Feed-forward reference (Qwen3 base) behind the same interface: one 'round' = the
    whole forward. Re-uses OuroLoop's loop/readout code with a Qwen3 layer."""

    kind = "ff"

    def __init__(self, name="qwen3-1.7b-base", device="cuda", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = MODEL_ROOT / name
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(path)
        lm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa")
        self.lm = lm.to(device).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.inner = self.lm.model
        self.cfg = self.lm.config
        self.T_train = 1
        self.n_layers = self.cfg.num_hidden_layers
        self.layers = list(self.inner.layers)
        self.head_dim = self.cfg.head_dim
        self.rope = sys.modules[type(self.inner).__module__].apply_rotary_pos_emb
        self.device = torch.device(device)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.bos = None
        self.gqa = self.cfg.num_key_value_heads != self.cfg.num_attention_heads

    def _layer(self, r, li, x, cos, sin, hooks, edit, kv=None):
        layer = self.layers[li]
        b, s, _ = x.shape
        a = layer.self_attn
        h = layer.input_layernorm(x)
        q = a.q_norm(a.q_proj(h).view(b, s, -1, self.head_dim)).transpose(1, 2)
        k = a.k_norm(a.k_proj(h).view(b, s, -1, self.head_dim)).transpose(1, 2)
        v = a.v_proj(h).view(b, s, -1, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, cos, sin)
        if self.gqa:
            rep = q.shape[1] // k.shape[1]
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        if kv is not None:
            o = kv.attend((r, li), q, k, v, a.scaling)
        else:
            extra = hooks.attn_mask(r, li) if hooks and hooks.attn_mask else None
            m, causal = _mask(s, x.device, extra)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=causal, scale=a.scaling)
        o = a.o_proj(o.transpose(1, 2).reshape(b, s, -1))
        if hooks and hooks.attn_out:
            o = hooks.attn_out(r, li, o)
        x = x + o
        mlp = layer.mlp
        u = layer.post_attention_layernorm(x)
        act = mlp.act_fn(mlp.gate_proj(u)) * mlp.up_proj(u)
        y = mlp.down_proj(act)
        dW = edit.get(li, r) if edit else None
        if dW is not None:
            y = y + (act.float() @ dW.T).to(y.dtype)
        if hooks and hooks.mlp_out:
            y = hooks.mlp_out(r, li, y)
        x = x + y
        if hooks and hooks.after_layer:
            x = hooks.after_layer(r, li, x)
        return x


def load(name, device="cuda", dtype=torch.bfloat16):
    if name.startswith("qwen"):
        return QwenFF(name, device, dtype)
    if name.startswith("ouro"):
        return OuroLoop(name, device, dtype)
    if name.startswith("huginn"):
        return HuginnLoop(name, device, dtype)
    raise ValueError(name)


# ----------------------------------------------------------------------------- batching


def pad_right(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, device=device)
    last = torch.tensor([len(s) - 1 for s in seqs], device=device)
    return ids, last


def round_logits_last(model, seqs, T, hooks=None, edit=None, rounds_keep=None):
    """Logits at each row's last real position for every round 1..T -> [T, B, V] (fp32)."""
    ids, last = pad_right(seqs, model.pad_id, model.device)
    rows = torch.arange(len(seqs), device=model.device)
    out = []
    for r, x in model.rounds(ids, T, hooks, edit):
        if rounds_keep is not None and r not in rounds_keep:
            continue
        if model.kind == "huginn":
            out.append(model.readout(x, (rows, last), r=r, hooks=hooks, edit=edit))
        else:
            out.append(model.readout(x, (rows, last)))
    return torch.stack(out)


def greedy_at_round(model, prompts, r, max_new=10, hooks=None, edit=None, stop_newline=True):
    """Greedy continuation of each prompt (token lists) using the r-round model.
    Full recompute per step; returns list of generated token lists."""
    seqs = [list(p) for p in prompts]
    gen = [[] for _ in prompts]
    done = [False] * len(prompts)
    nl = set()
    if stop_newline:
        for t in ["\n", "\n\n", ".\n"]:
            ids = model.tok(t, add_special_tokens=False).input_ids
            if len(ids) == 1:
                nl.add(ids[0])
    eos = {model.tok.eos_token_id} if model.tok.eos_token_id is not None else set()
    for _ in range(max_new):
        act = [i for i in range(len(seqs)) if not done[i]]
        if not act:
            break
        lg = round_logits_last(model, [seqs[i] for i in act], r, hooks, edit, rounds_keep={r})[0]
        nxt = lg.argmax(-1).tolist()
        for i, t in zip(act, nxt):
            seqs[i].append(t)
            gen[i].append(t)
            if t in eos or t in nl:
                done[i] = True
    return gen


@torch.no_grad()
def greedy_cached(model, prompts, r, max_new=10, edit=None, stop_newline=True):
    """Same output contract as greedy_at_round, with left padding and a KV cache."""
    dev = model.device
    n = len(prompts)
    P = max(len(p) for p in prompts)
    Lmax = P + max_new
    pad = torch.tensor([P - len(p) for p in prompts], device=dev)
    ids = torch.full((n, P), model.pad_id, dtype=torch.long, device=dev)
    for i, p in enumerate(prompts):
        ids[i, P - len(p):] = torch.tensor(p, device=dev)
    valid = torch.arange(Lmax, device=dev)[None] >= pad[:, None]
    kv = KV(valid)
    pos = (torch.arange(P, device=dev)[None] - pad[:, None]).clamp(min=0)
    nl = set()
    if stop_newline:
        for t in ["\n", "\n\n", ".\n"]:
            tt = model.tok(t, add_special_tokens=False).input_ids
            if len(tt) == 1:
                nl.add(tt[0])
    eos = {model.tok.eos_token_id} if model.tok.eos_token_id is not None else set()
    gen = [[] for _ in prompts]
    done = torch.zeros(n, dtype=torch.bool)
    cur, cur_pos = ids, pos
    for t in range(max_new):
        kv.start = 0 if t == 0 else P + t - 1
        x = None
        for _, x in model.rounds(cur, r, None, edit, pos=cur_pos, kv=kv):
            pass
        if model.kind == "huginn":
            lg = model.readout(x, None, r=r, edit=edit, kv=kv)[:, -1]
        else:
            lg = model.readout(x[:, -1])
        nxt = lg.argmax(-1)
        for i, tok in enumerate(nxt.tolist()):
            if not done[i]:
                gen[i].append(tok)
                if tok in eos or tok in nl:
                    done[i] = True
        if bool(done.all()):
            break
        cur = nxt[:, None]
        cur_pos = (P + t - pad)[:, None]
    return gen
