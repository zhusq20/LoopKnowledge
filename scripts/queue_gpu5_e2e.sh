#!/bin/bash
# 2E/1C on Ouro-1.4B: lr chosen by train-phrasing accuracy and known-fact retention only.
cd /mnt/disk2_from_server2/siqizhu4/LoopKnowledge
PY=~/.venvs/loopexp/bin/python
export CUDA_VISIBLE_DEVICES=5
$PY scripts/e2e_newfacts.py --model ouro-1.4b --lr 3e-5 --sup final --tag all-final-lr3e-5 > results/logs/e2e-ouro-all-final-lr3e-5.log 2>&1
$PY scripts/e2e_newfacts.py --model ouro-1.4b --lr 1e-5 --sup final --epochs 40 --tag all-final-lr1e-5 > results/logs/e2e-ouro-all-final-lr1e-5.log 2>&1
$PY scripts/e2e_newfacts.py --model ouro-1.4b --lr 3e-5 --sup all --tag all-allrounds-lr3e-5 > results/logs/e2e-ouro-all-allrounds-lr3e-5.log 2>&1
