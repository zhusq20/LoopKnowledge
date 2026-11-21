#!/bin/bash
# Phase 0 editor calibration on DEV hop1+hop2 cases. Selection reads only single-hop efficacy and neighbours.
cd /mnt/disk2_from_server2/siqizhu4/LoopKnowledge
PY=~/.venvs/loopexp/bin/python
run() { tag=$1; shift; $PY scripts/e2b_edit.py --model ouro-1.4b --split dev --groups hop1 hop2 --n 15 --read-rounds 4 --n-neighbors 24 --tag cal-$tag "$@" > results/logs/cal-$tag.log 2>&1; }
run L5-lr1e-4       --layers 5 --lr 1e-4 --steps 60
run L5-lr5e-4-c5e-4 --layers 5 --lr 5e-4 --linf 5e-4
run L456-lr1e-4     --layers 4 5 6 --lr 1e-4 --steps 60
run L456-lr5e-4-c2e-4 --layers 4 5 6 --lr 5e-4 --linf 2e-4
run L10-lr1e-4      --layers 10 --lr 1e-4 --steps 60
run L16-lr1e-4      --layers 16 --lr 1e-4 --steps 60
run L2-lr1e-4       --layers 2 --lr 1e-4 --steps 60
