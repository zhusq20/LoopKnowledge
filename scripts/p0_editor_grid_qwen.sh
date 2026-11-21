#!/bin/bash
# Phase 0 editor calibration for the feed-forward reference (DEV, single-hop metrics only).
cd /mnt/disk2_from_server2/siqizhu4/LoopKnowledge
PY=~/.venvs/loopexp/bin/python
run() { tag=$1; shift; $PY scripts/e2b_edit.py --model qwen3-1.7b-base --T 1 --split dev --groups hop1 hop2 --n 15 --read-rounds 1 --n-neighbors 24 --tag cal-$tag "$@" > results/logs/cal-qwen-$tag.log 2>&1; }
run L4-lr1e-4  --layers 4 --lr 1e-4 --steps 60
run L8-lr1e-4  --layers 8 --lr 1e-4 --steps 60
run L14-lr1e-4 --layers 14 --lr 1e-4 --steps 60
run L20-lr1e-4 --layers 20 --lr 1e-4 --steps 60
