#!/usr/bin/env python3
"""
批量训练ScanNet++场景 - 多GPU并行
使用5个GPU并行处理多个场景
"""

import os
import time
import subprocess
from pathlib import Path
from datetime import datetime
import json

# 场景配置 - 参考VoxelGS的划分
normal_scenes = ['0a184cf634', '13c3e046d7', '1d003b07bd', '260db9cf5a', '8be0cd3817', '6464461276', '8b5caf3398']
large_scenes = ['578511c8a9', '036bce3393', '6f1848d1e3', '281ba69af1']
larger_scenes = ['9460c8889d']

all_scenes = normal_scenes + large_scenes + larger_scenes

# GPU配置 - 使用5个GPU (0-4)
available_gpus = [0, 1, 2, 3, 4]

# 路径配置
exp_dir = "/root/autodl-tmp/Proj/GSDF/exp"
config_dir = "/root/autodl-tmp/Proj/GSDF/configs/scannetpp"
tag = "with_prior"

# 颜色输出
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_color(text, color=Colors.NC):
    print(f"{color}{text}{Colors.NC}")

def find_latest_checkpoint(scene_name):
    """查找场景的最新checkpoint"""
    scene_exp_dir = os.path.join(exp_dir, scene_name)
    if not os.path.exists(scene_exp_dir):
        return None
    
    ckpt_files = list(Path(scene_exp_dir).rglob("*.ckpt"))
    if not ckpt_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    return str(latest_ckpt)

def train_scene(scene_name, gpu_id):
    """训练单个场景"""
    print_color(f"\n{'='*60}", Colors.BLUE)
    print_color(f"开始训练场景: {scene_name} on GPU {gpu_id}", Colors.GREEN)
    print_color(f"{'='*60}", Colors.BLUE)
    
    config_file = os.path.join(config_dir, f"{scene_name}.yaml")
    
    if not os.path.exists(config_file):
        print_color(f"错误: 配置文件不存在 {config_file}", Colors.RED)
        return False
    
    # 检查是否有checkpoint
    latest_ckpt = find_latest_checkpoint(scene_name)
    
    cmd = [
        "python", "launch.py",
        "--exp_dir", exp_dir,
        "--config", config_file,
        "--gpu", str(gpu_id),
        "--train",
        "--eval",
        f"tag={tag}"
    ]
    
    if latest_ckpt:
        print_color(f"✓ 找到checkpoint: {latest_ckpt}", Colors.GREEN)
        print_color("从checkpoint恢复训练...", Colors.YELLOW)
        cmd.insert(-1, "--resume")
        cmd.insert(-1, latest_ckpt)
    else:
        print_color("⚠ 未找到checkpoint，从头开始训练...", Colors.YELLOW)
    
    print_color(f"命令: {' '.join(cmd)}", Colors.BLUE)
    
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 运行训练
    start_time = time.time()
    try:
        result = subprocess.run(cmd, env=env, check=True)
        elapsed = time.time() - start_time
        print_color(f"\n✓ 场景 {scene_name} 训练完成! 耗时: {elapsed/3600:.2f}小时", Colors.GREEN)
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print_color(f"\n✗ 场景 {scene_name} 训练失败! 耗时: {elapsed/3600:.2f}小时", Colors.RED)
        print_color(f"错误: {e}", Colors.RED)
        return False

def run_parallel_training():
    """并行训练多个场景"""
    print_color("\n" + "="*80, Colors.BLUE)
    print_color("GSDF ScanNet++ 批量训练脚本 (多GPU并行)", Colors.GREEN)
    print_color("="*80, Colors.BLUE)
    print_color(f"总场景数: {len(all_scenes)}", Colors.BLUE)
    print_color(f"可用GPU: {available_gpus}", Colors.BLUE)
    print_color(f"配置目录: {config_dir}", Colors.BLUE)
    print_color(f"实验目录: {exp_dir}", Colors.BLUE)
    print_color("="*80 + "\n", Colors.BLUE)
    
    # 记录开始时间
    overall_start = time.time()
    
    # 用于追踪GPU和场景的映射
    gpu_processes = {gpu: None for gpu in available_gpus}
    scene_queue = all_scenes.copy()
    completed_scenes = []
    failed_scenes = []
    
    print_color(f"待训练场景: {scene_queue}", Colors.BLUE)
    
    # 简化版本: 顺序执行，但可以手动分配GPU
    # 如果需要真正的并行，需要使用multiprocessing
    for i, scene in enumerate(scene_queue):
        gpu_id = available_gpus[i % len(available_gpus)]
        
        print_color(f"\n进度: [{i+1}/{len(scene_queue)}]", Colors.YELLOW)
        success = train_scene(scene, gpu_id)
        
        if success:
            completed_scenes.append(scene)
        else:
            failed_scenes.append(scene)
    
    # 打印总结
    overall_elapsed = time.time() - overall_start
    print_color("\n" + "="*80, Colors.BLUE)
    print_color("训练总结", Colors.GREEN)
    print_color("="*80, Colors.BLUE)
    print_color(f"总耗时: {overall_elapsed/3600:.2f}小时", Colors.BLUE)
    print_color(f"成功: {len(completed_scenes)}/{len(all_scenes)}", Colors.GREEN)
    print_color(f"失败: {len(failed_scenes)}/{len(all_scenes)}", Colors.RED if failed_scenes else Colors.GREEN)
    
    if completed_scenes:
        print_color("\n成功的场景:", Colors.GREEN)
        for scene in completed_scenes:
            print_color(f"  ✓ {scene}", Colors.GREEN)
    
    if failed_scenes:
        print_color("\n失败的场景:", Colors.RED)
        for scene in failed_scenes:
            print_color(f"  ✗ {scene}", Colors.RED)
    
    print_color("="*80 + "\n", Colors.BLUE)
    
    # 保存结果
    summary = {
        'total_time_hours': overall_elapsed / 3600,
        'completed': completed_scenes,
        'failed': failed_scenes,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(exp_dir, 'training_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print_color(f"训练总结已保存到: {summary_file}", Colors.BLUE)
    
    return len(failed_scenes) == 0

if __name__ == '__main__':
    import sys
    success = run_parallel_training()
    sys.exit(0 if success else 1)
