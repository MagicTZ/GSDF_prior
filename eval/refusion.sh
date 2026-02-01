#!/bin/bash
# tsdf_refusion.sh - 批量评估与汇总脚本
# 
# 功能：
# 1. 遍历 Proj/GSDF/exp 下的所有实验，自动进行 Clipping 和 Evaluation
# 2. 自动汇总所有场景的指标到 summary.csv

# 设置GSDF项目路径到PYTHONPATH
export PYTHONPATH="/root/autodl-tmp/Proj/GSDF:$PYTHONPATH"

# ================= 配置区域 =================

# 1. 路径配置
EXP_ROOT="/root/autodl-tmp/Proj/GSDF/exp"
DATASET_ROOT="/root/autodl-tmp/Proj/GS-Reconstruction/Data/ScanNetpp"

# 2. 目标 Mesh 文件名模式
# 我们要找的是 with_prior_new@.../save/it30000-mc1024_original.ply
TARGET_MESH_NAME="it30000-mc1024_original.ply"

# 3. TSDF参数 (如果 SKIP_REFUSE=true 则不生效)
VOXEL_SIZE=0.5
DEPTH_TRUNC=5.0
SAMPLE_INTERVAL=10
DOWNSAMPLE=2

# 4. 裁剪与评估参数
CLIP_MARGIN=0.03       # 裁剪边界扩展（米）
USE_CLIP=true          # 是否启用基于GT BBox的裁剪

# 5. Refusion 开关
SKIP_REFUSE=true       # 是否跳过 TSDF Refusion（只做 Clip + Eval）

# ===========================================

# 切换到GSDF目录
cd /root/autodl-tmp/Proj/GSDF

echo "开始批量评估..."
echo "实验根目录: $EXP_ROOT"

# 遍历所有实验目录
for exp_dir in "$EXP_ROOT"/*; do
    if [ ! -d "$exp_dir" ]; then continue; fi
    
    # 获取实验ID
    exp_id=$(basename "$exp_dir")
    
    # 忽略 evaluation_results 等非实验目录
    if [ "$exp_id" == "evaluation_results" ]; then continue; fi
    
    echo "--------------------------------------------------"
    echo "处理实验: $exp_id"
    
    # 1. 查找 Pred Mesh
    found_meshes=$(find "$exp_dir" -name "$TARGET_MESH_NAME" | grep "with_prior_new")
    
    if [ -z "$found_meshes" ]; then
        echo "[Skip] 未找到目标 Mesh: $TARGET_MESH_NAME"
        continue
    fi
    
    # 2. 查找对应的 GT Mesh
    gt_mesh="$DATASET_ROOT/$exp_id/mesh.ply"
    data_root="$DATASET_ROOT/$exp_id"
    
    if [ ! -f "$gt_mesh" ]; then
        echo "[Skip] 未找到对应的 GT Mesh: $gt_mesh"
        continue
    fi
    
    # 对每个找到的 Mesh 进行评估
    while IFS= read -r mesh_path; do
        if [ -z "$mesh_path" ]; then continue; fi
        
        echo "  Found Mesh: $mesh_path"
        
        # 定义输出路径
        output_dir=$(dirname "$mesh_path")
        output_path="${output_dir}/it30000-mc1024_processed.ply"
        
        # 检查是否已生成 stats 文件，如果已存在则跳过 (可选，为了速度)
        stats_file="${output_dir}/it30000-mc1024_processed_stats.json"
        # if [ -f "$stats_file" ]; then
        #     echo "  [Skip] 结果已存在: $stats_file"
        #     continue
        # fi
        
        # 构建命令
        CMD="python eval/eval_refuse_mesh.py \
            --mesh \"$mesh_path\" \
            --data_root \"$data_root\" \
            --output \"$output_path\" \
            --gt \"$gt_mesh\" \
            --voxel_size $VOXEL_SIZE \
            --depth_trunc $DEPTH_TRUNC \
            --sample_interval $SAMPLE_INTERVAL \
            --downsample $DOWNSAMPLE"

        if [ "$USE_CLIP" = true ]; then
            CMD="$CMD --use_clip --clip_margin $CLIP_MARGIN"
        fi
        
        if [ "$SKIP_REFUSE" = true ]; then
            CMD="$CMD --skip_refuse"
        fi
        
        # 执行命令
        echo "  Running evaluation..."
        eval $CMD
        
    done <<< "$found_meshes"
    
done

echo "--------------------------------------------------"
echo "所有实验评估完成！"

# ================= 汇总结果 =================

SUMMARY_FILE="$EXP_ROOT/evaluation_results/summary.csv"
mkdir -p "$(dirname "$SUMMARY_FILE")"

echo "正在生成汇总报表..."

# 使用嵌入的 Python 脚本进行汇总
python3 -c "
import os
import json
import glob
import pandas as pd
from pathlib import Path

exp_root = '$EXP_ROOT'
output_file = '$SUMMARY_FILE'

# 查找所有 stats 文件 (processed, clipped, tsdf)
patterns = [
    '*_processed_stats.json', 
    '*_clipped_stats.json', 
    '*_tsdf_stats.json'
]
files = []
for p in patterns:
    files.extend(glob.glob(os.path.join(exp_root, '*', 'with_prior_new@*', 'save', p)))

results = []
seen_scenes = set() # 每个场景只取最新的一个结果

# 按时间倒序排序文件，确保优先取最新的
# 假设路径包含时间戳 .../with_prior_new@TIMESTAMP/...
try:
    files.sort(key=lambda x: str(Path(x).parent.parent), reverse=True)
except:
    pass

for fpath in files:
    try:
        # 提取 Scene ID (倒数第4层目录)
        scene_id = Path(fpath).parts[-4]
        
        # 如果该场景已处理过（已取到最新的），则跳过
        if scene_id in seen_scenes: continue
        seen_scenes.add(scene_id)
        
        with open(fpath, 'r') as f:
            data = json.load(f)
        
        acc = data.get('Accuracy', 0)
        comp = data.get('Completeness', 0)
        chamfer = (acc + comp) / 2
        
        results.append({
            'Scene ID': scene_id,
            'Accuracy': acc,
            'Completeness': comp,
            'Chamfer': chamfer,
            'Precision': data.get('Precision', 0),
            'Recall': data.get('Recall', 0),
            'F-score': data.get('F-score', 0)
        })
    except Exception as e:
        print(f'Error reading {fpath}: {e}')

if results:
    df = pd.DataFrame(results)
    # 按 Scene ID 排序
    df = df.sort_values('Scene ID')
    
    # 计算平均值
    mean_row = df.mean(numeric_only=True)
    mean_row['Scene ID'] = 'AVERAGE'
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    # 设置列顺序
    cols = ['Scene ID', 'Accuracy', 'Completeness', 'Chamfer', 'Precision', 'Recall', 'F-score']
    df = df[cols]
    
    # 保存 CSV
    df.to_csv(output_file, index=False)
    
    # 打印表格
    print('\n' + df.to_string(index=False))
    print(f'\n汇总已保存至: {output_file}')
else:
    print('未找到有效的评估结果文件。')
"
