"""MQuAKE-CF-3k-v2 -> single-hop facts, two-hop chains (nested noun-phrase cloze), splits.

Split: case-level, sha256("lk-split-v1:<case_id>") % 10 < 3 -> dev (about 30%), else test.
A fact or chain gets the split of the first case (lowest case_id) it appears in.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "mquake"

# noun phrase "the R of X" for each relation label (used as the first hop inside a nested cloze)
NOUN = {
    "country of origin": "the country of origin of {}",
    "sport": "the sport played by {}",
    "country of citizenship": "the country of citizenship of {}",
    "capital": "the capital of {}",
    "founded by": "the founder of {}",
    "continent": "the continent where {} is located",
    "official language": "the official language of {}",
    "religion or worldview": "the religion of {}",
    "position played on team / speciality": "the position played by {}",
    "author": "the author of {}",
    "place of birth": "the birthplace of {}",
    "creator": "the creator of {}",
    "spouse": "the spouse of {}",
    "performer": "the performer of {}",
    "director / manager": "the director of {}",
    "developer": "the developer of {}",
    "head of state": "the head of state of {}",
    "chief executive officer": "the chief executive officer of {}",
    "headquarters location": "the city where {} is headquartered",
    "head of government": "the head of government of {}",
    "genre": "the music genre of {}",
    "work location": "the city where {} worked",
    "original broadcaster": "the original broadcaster of {}",
    "educated at": "the university where {} was educated",
    "employer": "the employer of {}",
    "manufacturer": "the manufacturer of {}",
    "notable work": "the most famous work of {}",
    "languages spoken, written or signed": "the language spoken by {}",
    "place of death": "the city where {} died",
    "chairperson": "the chairperson of {}",
    "head coach": "the head coach of {}",
    "occupation": "the occupation of {}",
    "child": "the child of {}",
    "location of formation": "the city where {} was founded",
    "language of work or name": "the language in which {} was written",
    "officeholder": "the current {}",
    "original language of film or TV show": "the original language of {}",
}


def split_of(case_id: int) -> str:
    h = int(hashlib.sha256(f"lk-split-v1:{case_id}".encode()).hexdigest(), 16)
    return "dev" if h % 10 < 3 else "test"


def load_cases(name="MQuAKE-CF-3k-v2.json"):
    return json.load(open(DATA / name))


def template_of(cloze: str, subject: str) -> str:
    assert subject in cloze, (cloze, subject)
    return cloze.replace(subject, "{}", 1)


def cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def nested(t2: str, r1_label: str, s1: str) -> str:
    np_ = NOUN[r1_label].format(s1)
    out = t2.replace("{}", np_)
    if t2.startswith("{}"):
        out = cap(out)
    return out


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s.lower())
    return " ".join(s.split())


def alias_list(answer, aliases):
    out = []
    for a in [answer] + list(aliases or []):
        n = norm(a)
        if n and n not in out:
            out.append(n)
    return out


def match(gen_text: str, aliases: list[str]) -> bool:
    g = " " + norm(gen_text) + " "
    return any((" " + a + " ") in g for a in aliases)


def facts_and_chains(cases):
    """Returns facts {key: fact} and chains {key: chain} built from the ORIGINAL chains.
    fact = subject, rel, rel_label, obj, cloze, template, aliases, split, case_ids
    chain (two-hop) = s1, r1, bridge, r2, answer, nested cloze, aliases, fact keys."""
    facts, chains = {}, {}
    for c in sorted(cases, key=lambda c: c["case_id"]):
        sp = split_of(c["case_id"])
        tl, tr = c["orig"]["triples_labeled"], c["orig"]["triples"]
        hops = c["single_hops"]
        keys = []
        for (s, rl, o), (_, rid, _), h in zip(tl, tr, hops):
            k = (s, rid)
            keys.append(k)
            if k not in facts:
                facts[k] = {
                    "key": f"{s}||{rid}",
                    "subject": s,
                    "rel": rid,
                    "rel_label": rl,
                    "obj": o,
                    "cloze": h["cloze"],
                    "template": template_of(h["cloze"], s),
                    "aliases": alias_list(h["answer"], h["answer_alias"]),
                    "split": sp,
                    "case_ids": [],
                }
            facts[k]["case_ids"].append(c["case_id"])
        for i in range(len(tl) - 1):
            (s1, rl1, b), (_, rl2, a) = tl[i], tl[i + 1]
            k = (s1, tr[i][1], tr[i + 1][1])
            if k in chains:
                chains[k]["case_ids"].append(c["case_id"])
                continue
            t2 = facts[keys[i + 1]]["template"]
            chains[k] = {
                "key": f"{s1}||{tr[i][1]}||{tr[i + 1][1]}",
                "s1": s1,
                "r1": tr[i][1],
                "r1_label": rl1,
                "bridge": b,
                "r2": tr[i + 1][1],
                "r2_label": rl2,
                "answer": a,
                "cloze": nested(t2, rl1, s1),
                "context_cloze": facts[keys[i]]["cloze"] + " " + b + ". " + nested(t2, rl1, s1),
                "aliases": facts[keys[i + 1]]["aliases"],
                "hop1": facts[keys[i]]["key"],
                "hop2": facts[keys[i + 1]]["key"],
                "t2": t2,
                "split": sp,
                "case_ids": [c["case_id"]],
                "pos_in_case": i,
            }
    return facts, chains


def three_hop_chains(cases):
    """(s1, r1, r2, r3) chains from cases with >= 3 hops (for 3A parametric chains)."""
    out = {}
    for c in sorted(cases, key=lambda c: c["case_id"]):
        tl, tr, hops = c["orig"]["triples_labeled"], c["orig"]["triples"], c["single_hops"]
        for i in range(len(tl) - 2):
            k = (tl[i][0], tr[i][1], tr[i + 1][1], tr[i + 2][1])
            if k in out:
                continue
            t3 = template_of(hops[i + 2]["cloze"], tl[i + 2][0])
            inner = NOUN[tl[i][1]].format(tl[i][0])
            mid = NOUN[tl[i + 1][1]].format(inner)
            cl = t3.replace("{}", mid)
            if t3.startswith("{}"):
                cl = cap(cl)
            out[k] = {
                "key": "||".join(k),
                "cloze": cl,
                "aliases": alias_list(hops[i + 2]["answer"], hops[i + 2]["answer_alias"]),
                "hops": [f"{tl[j][0]}||{tr[j][1]}" for j in range(i, i + 3)],
                "split": split_of(c["case_id"]),
            }
    return out


if __name__ == "__main__":
    cases = load_cases()
    f, ch = facts_and_chains(cases)
    t3 = three_hop_chains(cases)
    print(len(f), "facts;", len(ch), "two-hop chains;", len(t3), "three-hop chains")
    print(sum(x["split"] == "dev" for x in f.values()), "dev facts")
    import random

    for x in random.Random(0).sample(list(ch.values()), 6):
        print(x["cloze"], "->", x["answer"], "| ctx:", x["context_cloze"])
    for x in random.Random(0).sample(list(t3.values()), 3):
        print(x["cloze"], "->", x["aliases"][0])
