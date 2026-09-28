#!/bin/bash
cd /home/user_beatrizalmeida/elastic_weight_streaming

echo "Iniciando a busca pelo ponto doce (lr=1e-4)..."

CUDA_VISIBLE_DEVICES=2 nohup /usr/bin/python3 -m glod amq --model Qwen/Qwen3-4B --device cuda:0 --rank 16 --lr-lowrank 1e-4 --kinds kl --steps 1500 --tag-suffix _sweetspot > var/logs/amq/qwen3-4b_r16_1e4_kl.log 2>&1 &
PID_KL=$!

CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup /usr/bin/python3 -m glod amq --model Qwen/Qwen3-4B --device cuda:0 --rank 16 --lr-lowrank 1e-4 --kinds flip --steps 1500 --tag-suffix _sweetspot > var/logs/amq/qwen3-4b_r16_1e4_flip.log 2>&1 &
PID_FLIP=$!

echo "Variante KL lanÃ§ada na GPU2 (PID: $PID_KL)"
echo "Variante flip lanÃ§ada na GPU1 (PID: $PID_FLIP)"
echo "Acompanhe os logs em:"
echo " - var/logs/amq/qwen3-4b_r16_1e4_kl.log"
echo " - var/logs/amq/qwen3-4b_r16_1e4_flip.log"
