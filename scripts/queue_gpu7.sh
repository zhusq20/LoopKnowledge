#!/bin/bash
# GPU-7 queue: feed-forward reference 2B (Qwen L20 selected on dev, L4 classic early layer), then Huginn 1B.
cd /mnt/disk2_from_server2/siqizhu4/LoopKnowledge
PY=~/.venvs/loopexp/bin/python
export CUDA_VISIBLE_DEVICES=7
for L in 20 4; do
  $PY scripts/e2b_edit.py --model qwen3-1.7b-base --T 1 --layers $L --lr 1e-4 --steps 60 --split test --groups hop1 hop2 both --n 120 --read-rounds 1 --tag test-L$L > results/logs/e2b-qwen-L$L.log 2>&1
done
U="1,2,3,4,5,6,7,8;9,10,11,12,13,14,15,16;17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32;32;1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
$PY scripts/e2b_edit.py --model huginn-0125 --T 32 --layers 1 2 --lr 1e-5 --steps 60 --split test --groups hop1 hop2 --n 60 --read-rounds 4 8 16 32 --unroll "$U" --tag test-1B-core12 > results/logs/e1b-huginn-core12.log 2>&1
$PY scripts/e2b_edit.py --model huginn-0125 --T 32 --layers prelude:0 prelude:1 --lr 1e-5 --steps 60 --split test --groups hop1 hop2 --n 60 --read-rounds 4 8 16 32 --tag test-1B-prelude > results/logs/e1b-huginn-prelude.log 2>&1
$PY scripts/e2b_edit.py --model huginn-0125 --T 32 --layers coda:0 coda:1 --lr 1e-5 --steps 60 --split test --groups hop1 hop2 --n 60 --read-rounds 4 8 16 32 --tag test-1B-coda > results/logs/e1b-huginn-coda.log 2>&1
