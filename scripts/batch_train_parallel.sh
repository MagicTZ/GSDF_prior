#!/bin/bash
# 真正的并行训练脚本 - 使用后台进程同时训练多个场景

exp_dir=/root/autodl-tmp/Proj/GSDF/exp
config_dir=/root/autodl-tmp/Proj/GSDF/configs/scannetpp
tag=with_prior_new

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================"
echo "GSDF ScanNet++ 并行训练脚本"
echo -e "========================================${NC}"

# 场景列表 (参考VoxelGS)
normal_scenes=("0a184cf634" "13c3e046d7" "1d003b07bd" "260db9cf5a" "8be0cd3817" "6464461276" "8b5caf3398")
large_scenes=("578511c8a9" "036bce3393" "6f1848d1e3" "281ba69af1")
larger_scenes=("9460c8889d")

# TODO: 所有场景
all_scenes=("${normal_scenes[@]}" "${large_scenes[@]}" "${larger_scenes[@]}")
all_scenes=("9460c8889d" "281ba69af1")

# TODO: 可用GPU (使用0-4, 共5个)
gpus=(0 1)
num_gpus=${#gpus[@]}

echo -e "${BLUE}总场景数: ${#all_scenes[@]}${NC}"
echo -e "${BLUE}可用GPU: ${gpus[@]}${NC}"
echo ""

# 函数: 训练单个场景
train_scene() {
    local scene=$1
    local gpu=$2
    local config="${config_dir}/${scene}.yaml"
    
    echo -e "${GREEN}[GPU ${gpu}] 开始训练场景: ${scene}${NC}"
    
    # 检查配置文件
    if [ ! -f "$config" ]; then
        echo -e "${RED}[GPU ${gpu}] 错误: 配置文件不存在 ${config}${NC}"
        return 1
    fi
    
    # 查找最新checkpoint
    latest_ckpt=$(find ${exp_dir}/${scene} -name "*.ckpt" -type f 2>/dev/null | sort -r | head -1)

    # 构建命令
    # 注意: CUDA_VISIBLE_DEVICES=${gpu} 会让进程只看到指定GPU，该GPU在进程内被映射为0
    # 所以 --gpu 0 是正确的，它使用的是CUDA_VISIBLE_DEVICES指定的那个GPU
    cmd="source /root/autodl-tmp/Proj/.venv_cuda128/bin/activate \
        && CUDA_VISIBLE_DEVICES=${gpu} \
        python launch.py \
        --exp_dir ${exp_dir} \
        --config ${config} \
        --gpu 0 \
        --train \
        --eval \
        tag=${tag}"
    
    # if [ -n "$latest_ckpt" ]; then
    #     echo -e "${GREEN}[GPU ${gpu}] 找到checkpoint: ${latest_ckpt}${NC}"
    #     cmd="${cmd} --resume ${latest_ckpt}"
    # else
    #     echo -e "${YELLOW}[GPU ${gpu}] 未找到checkpoint，从头训练${NC}"
    # fi

    echo -e "${YELLOW}[GPU ${gpu}] ，从头训练${NC}"

    
    # 运行训练，输出重定向到日志
    log_file="${exp_dir}/${scene}/training.log"
    mkdir -p "${exp_dir}/${scene}"
    
    echo -e "${BLUE}[GPU ${gpu}] 命令: ${cmd}${NC}"
    echo -e "${BLUE}[GPU ${gpu}] 日志: ${log_file}${NC}"
    
    eval ${cmd} > ${log_file} 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[GPU ${gpu}] ✓ 场景 ${scene} 训练完成${NC}"
    else
        echo -e "${RED}[GPU ${gpu}] ✗ 场景 ${scene} 训练失败 (exit code: ${exit_code})${NC}"
    fi
    
    return $exit_code
}

# 导出函数供并行使用
export -f train_scene
export exp_dir config_dir tag RED GREEN YELLOW BLUE NC

# 并行训练 - 每次最多同时运行num_gpus个任务
scene_idx=0
pids=()
scene_gpu_map=()

for scene in "${all_scenes[@]}"; do
    gpu_idx=$((scene_idx % num_gpus))
    gpu=${gpus[$gpu_idx]}
    
    # 在后台启动训练
    train_scene "$scene" "$gpu" &
    pid=$!
    pids+=($pid)
    scene_gpu_map+=("${scene}:GPU${gpu}:PID${pid}")
    
    echo -e "${YELLOW}[调度] 场景 ${scene} -> GPU ${gpu} (PID: ${pid})${NC}"
    
    # 如果已经启动了num_gpus个进程，等待其中一个完成
    if [ $(( (scene_idx + 1) % num_gpus )) -eq 0 ] && [ $scene_idx -lt $((${#all_scenes[@]} - 1)) ]; then
        echo -e "${YELLOW}等待当前批次完成...${NC}"
        wait -n  # 等待任意一个后台任务完成
    fi
    
    scene_idx=$((scene_idx + 1))
    
    # 避免同时启动太多进程，稍微延迟
    sleep 5
done

# 等待所有任务完成
echo -e "${YELLOW}等待所有训练任务完成...${NC}"
wait

echo ""
echo -e "${BLUE}========================================"
echo "所有训练任务已完成"
echo -e "========================================${NC}"

# 统计结果
echo ""
echo -e "${BLUE}场景-GPU映射:${NC}"
for mapping in "${scene_gpu_map[@]}"; do
    echo "  $mapping"
done

echo ""
echo -e "${GREEN}✓ 批量训练完成!${NC}"
echo -e "${BLUE}查看各场景日志: ${exp_dir}/[scene_name]/training.log${NC}"
