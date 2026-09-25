#!/bin/bash
# Script para buscar o 'ponto doce' da taxa de aprendizado do adaptador de baixo posto (r16).
# Tentamos 1e-3 (explodiu) e 1e-5 (ficou engessado e nÃ£o superou o baseline).
# Agora testamos 1e-4 para ver se conseguimos liberdade de direÃ§Ã£o efetiva sem divergir.

cd /home/user_beatrizalmeida/elastic_weight_streaming

echo "Iniciando a busca pelo ponto doce (lr=1e-4)..."

# Como o cache 'dense_targets_selfgen.pt' jÃ¡ foi gerado na rodada anterior pelo Claude, 
# podemos lanÃ§ar ambas as variantes simultaneamente sem medo de gargalo na memÃ³ria/cpu!

# LanÃ§a a variante KL na GPU2
CUDA_VISIBLE_DEVICES=2 nohup /usr/bin/python3 -m glod amq --model Qwen/Qwen3-4B --device cuda:0 --rank 16 --lr-lowrank 1e-4 --kinds kl --steps 1500 --tag-suffix _sweetspot > var/logs/amq/qwen3-4b_r16_1e4_kl.log 2>&1 &
PID_KL=$!

# LanÃ§a a variante flip na GPU1
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup /usr/bin/python3 -m glod amq --model Qwen/Qwen3-4B --device cuda:0 --rank 16 --lr-lowrank 1e-4 --kinds flip --steps 1500 --tag-suffix _sweetspot > var/logs/amq/qwen3-4b_r16_1e4_flip.log 2>&1 &
PID_FLIP=$!

echo "Variante KL lanÃ§ada na GPU2 (PID: $PID_KL)"
echo "Variante flip lanÃ§ada na GPU1 (PID: $PID_FLIP)"
echo "Acompanhe os logs em:"
echo " - var/logs/amq/qwen3-4b_r16_1e4_kl.log"
echo " - var/logs/amq/qwen3-4b_r16_1e4_flip.log"
