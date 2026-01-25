#!/bin/bash
# 智能训练脚本 - 自动检测并恢复 checkpoint

exp_dir=./exp
config=configs/scannetpp/0a184cf634.yaml
gpu=0
tag=with_prior

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "GSDF ScanNet++ 智能训练脚本"
echo "========================================"

# 查找最新的 checkpoint
latest_ckpt=$(find ${exp_dir}/0a184cf634 -name "*.ckpt" -type f 2>/dev/null | sort -r | head -1)

if [ -n "$latest_ckpt" ]; then
    echo -e "${GREEN}✓ 找到 checkpoint: ${latest_ckpt}${NC}"
    echo "从 checkpoint 恢复训练..."
    
    python launch.py \
        --exp_dir ${exp_dir} \
        --config ${config} \
        --gpu ${gpu} \
        --train \
        --eval \
        --resume ${latest_ckpt} \
        tag=${tag} 
else
    echo -e "${YELLOW}⚠ 未找到 checkpoint${NC}"
    echo "从头开始训练..."
    
    python launch.py \
        --exp_dir ${exp_dir} \
        --config ${config} \
        --gpu ${gpu} \
        --train \
        --eval \
        tag=${tag} 
fi

echo ""
echo "========================================"
echo "训练完成或中断"
echo "========================================"
