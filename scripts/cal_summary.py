"""Editor calibration summary: single-hop efficacy and neighbour retention only (no multi-hop)."""
import json, sys, glob
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parents[1] / "results" / "e2b" / sys.argv[1]
for d in sorted(glob.glob(str(root / "cal-*"))):
    f = Path(d) / "cases.jsonl"
    if not f.exists():
        continue
    R = [json.loads(l) for l in open(f)]
    if not R:
        continue
    T = max(R[0]["after"]["edit_0"], key=int)
    eff = np.mean([all(v[T]["new"] for k, v in r["after"].items() if k.startswith("edit_")) for r in R])
    nb = np.mean([r["neigh_after"][T] for r in R])
    # other hop of the same case that was NOT edited (same-chain neighbour)
    keep = [r["after"]["old_hop_1"][T]["old"] for r in R if r["group"] == "hop1" and r["before"]["old_hop_1"][T]["old"]]
    keep += [r["after"]["old_hop_0"][T]["old"] for r in R if r["group"] == "hop2" and r["before"]["old_hop_0"][T]["old"]]
    pr = [all(v[T]["new"] for k, v in r["after"].items() if k.startswith("para_")) for r in R if "para_0" in r["after"]]
    st = np.mean([r["edit_info"]["steps"] for r in R]); nm = np.mean([sum(r["edit_info"]["norms"].values()) for r in R])
    print(f"{Path(d).name:28s} n={len(R):3d} efficacy@{T}={eff:.2f} neighbours={nb:.2f} other_hop_kept={np.mean(keep) if keep else float('nan'):.2f} para={np.mean(pr) if pr else float('nan'):.2f}(n={len(pr)}) steps={st:.1f} |dW|={nm:.2f}")
