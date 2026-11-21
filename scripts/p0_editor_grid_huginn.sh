#!/bin/bash
# Phase 0 editor calibration for Huginn (DEV hop1+hop2, single-hop metrics only). Regions for 1B.
# Huginn MLP out-proj std ~0.0043 (Ouro 0.021) and the core runs 32 times, so lr is 10-30x below Ouro's 1e-4.
cd /mnt/disk2_from_server2/siqizhu4/LoopKnowledge
PY=~/.venvs/loopexp/bin/python
run() { tag=$1; shift; $PY scripts/e2b_edit.py --model huginn-0125 --T 32 --split dev --groups hop1 hop2 --n 12 --read-rounds 32 --n-neighbors 24 --tag cal-$tag "$@" > results/logs/cal-huginn-$tag.log 2>&1; }
run core12-lr1e-5    --layers 1 2 --lr 1e-5 --steps 60
run core12-lr3e-6    --layers 1 2 --lr 3e-6 --steps 80
run prelude-lr1e-5   --layers prelude:0 prelude:1 --lr 1e-5 --steps 60
run coda-lr1e-5      --layers coda:0 coda:1 --lr 1e-5 --steps 60
