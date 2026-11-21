#!/bin/bash
# 1B on Ouro-1.4B (no prelude/coda): same editor, lower / middle / upper single layer. Test split.
cd /mnt/disk2_from_server2/siqizhu4/LoopKnowledge
PY=~/.venvs/loopexp/bin/python
for L in 4 12 20; do
  $PY scripts/e2b_edit.py --model ouro-1.4b --layers $L --lr 1e-4 --steps 60 --split test --groups hop1 hop2 --n 60 --read-rounds 1 2 3 4 --unroll "1;2;3;4" --tag test-1B-L$L > results/logs/e1b-ouro-L$L.log 2>&1
done
