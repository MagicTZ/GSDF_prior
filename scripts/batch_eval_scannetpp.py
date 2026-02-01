#!/usr/bin/env python3
"""
批量评估ScanNet++场景的mesh
参考VoxelGS的评估标准
"""

import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加eval目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'eval'))
from eval_gsdf_mesh import evaluate_gsdf_mesh

# 场景配置 - 与VoxelGS保持一致
normal_scenes = ['0a184cf634', '13c3e046d7', '1d003b07bd', '260db9cf5a', '8be0cd3817', '6464461276', '8b5caf3398']
large_scenes = ['578511c8a9', '036bce3393', '6f1848d1e3', '281ba69af1']
larger_scenes = ['9460c8889d']

all_scenes = normal_scenes + large_scenes + larger_scenes

# 路径配置
exp_dir = "/root/autodl-tmp/Proj/GSDF/exp"
data_root = "/root/autodl-tmp/Proj/GS-Reconstruction/Data/ScanNetpp"

# 评估参数 - 参考VoxelGS eval_recon.py
EVAL_PARAMS = {
    'threshold': 0.05,  # 5cm threshold for precision/recall
    'down_sample': 0.02,  # 2cm voxel downsampling
}

# 颜色输出
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def print_color(text, color=Colors.NC):
    print(f"{color}{text}{Colors.NC}")

def find_latest_mesh(scene_name, exp_dir):
    """查找场景最新的mesh文件"""
    scene_exp_dir = os.path.join(exp_dir, scene_name)
    
    # 查找_original.ply文件 (转换到原始坐标系的mesh)
    pattern = os.path.join(scene_exp_dir, "*/save/*_original.ply")
    mesh_files = glob.glob(pattern)
    
    if not mesh_files:
        print_color(f"  ⚠ 未找到mesh文件 (pattern: {pattern})", Colors.YELLOW)
        return None
    
    # 如果有多个文件，优先选择迭代次数最高的
    # 文件名格式：it30000-mc1024_original.ply
    def get_iteration(filepath):
        filename = os.path.basename(filepath)
        if filename.startswith('it'):
            try:
                return int(filename.split('-')[0][2:])  # 提取it后面的数字
            except:
                pass
        return 0
    
    # 按迭代次数排序，选择最高的
    latest_mesh = max(mesh_files, key=lambda x: (get_iteration(x), os.path.getmtime(x)))
    return latest_mesh

def find_gt_mesh(scene_name, data_root):
    """查找GT mesh"""
    gt_mesh = os.path.join(data_root, scene_name, "mesh.ply")
    if os.path.exists(gt_mesh):
        return gt_mesh
    else:
        print_color(f"  ⚠ GT mesh不存在: {gt_mesh}", Colors.YELLOW)
        return None

def evaluate_scene(scene_name):
    """评估单个场景"""
    print_color(f"\n{'='*60}", Colors.BLUE)
    print_color(f"评估场景: {scene_name}", Colors.GREEN)
    print_color(f"{'='*60}", Colors.BLUE)
    
    # 查找预测mesh
    pred_mesh = find_latest_mesh(scene_name, exp_dir)
    if pred_mesh is None:
        print_color(f"✗ 场景 {scene_name}: 未找到预测mesh", Colors.RED)
        return None
    
    print_color(f"预测mesh: {pred_mesh}", Colors.BLUE)
    
    # 查找GT mesh
    gt_mesh = find_gt_mesh(scene_name, data_root)
    if gt_mesh is None:
        print_color(f"✗ 场景 {scene_name}: 未找到GT mesh", Colors.RED)
        return None
    
    print_color(f"GT mesh: {gt_mesh}", Colors.BLUE)
    
    # 输出目录
    output_dir = os.path.dirname(pred_mesh)
    
    # 运行评估
    try:
        metrics = evaluate_gsdf_mesh(
            pred_mesh_path=pred_mesh,
            gt_mesh_path=gt_mesh,
            threshold=EVAL_PARAMS['threshold'],
            down_sample=EVAL_PARAMS['down_sample'],
            show_errormap=False,  # 禁用错误地图以避免Open3D崩溃
            output_dir=output_dir
        )
        
        if metrics:
            print_color(f"✓ 场景 {scene_name} 评估完成", Colors.GREEN)
            metrics['scene'] = scene_name
            metrics['pred_mesh'] = pred_mesh
            return metrics
        else:
            print_color(f"✗ 场景 {scene_name} 评估失败", Colors.RED)
            return None
            
    except Exception as e:
        print_color(f"✗ 场景 {scene_name} 评估出错: {e}", Colors.RED)
        return None

def run_batch_evaluation():
    """批量评估所有场景"""
    print_color("\n" + "="*80, Colors.BLUE)
    print_color("GSDF ScanNet++ 批量评估脚本", Colors.GREEN)
    print_color("="*80, Colors.BLUE)
    print_color(f"总场景数: {len(all_scenes)}", Colors.BLUE)
    print_color(f"实验目录: {exp_dir}", Colors.BLUE)
    print_color(f"数据目录: {data_root}", Colors.BLUE)
    print_color(f"评估参数: threshold={EVAL_PARAMS['threshold']}m, downsample={EVAL_PARAMS['down_sample']}m", Colors.BLUE)
    print_color("="*80 + "\n", Colors.BLUE)
    
    all_metrics = []
    success_count = 0
    
    for i, scene in enumerate(all_scenes):
        print_color(f"\n进度: [{i+1}/{len(all_scenes)}]", Colors.YELLOW)
        
        metrics = evaluate_scene(scene)
        
        if metrics:
            all_metrics.append(metrics)
            success_count += 1
    
    # 生成总结报告
    print_color("\n" + "="*80, Colors.BLUE)
    print_color("评估总结", Colors.GREEN)
    print_color("="*80, Colors.BLUE)
    
    if not all_metrics:
        print_color("没有成功评估的场景!", Colors.RED)
        return False
    
    # 创建DataFrame
    df = pd.DataFrame(all_metrics)
    
    # 按场景类型分组
    df['scene_type'] = df['scene'].apply(lambda x: 
        'normal' if x in normal_scenes else 
        'large' if x in large_scenes else 'larger')
    
    # 打印详细结果
    print_color(f"\n成功评估: {success_count}/{len(all_scenes)}", Colors.GREEN)
    print_color("\n各场景结果:", Colors.BLUE)
    print("-" * 100)
    print(f"{'Scene':<15} {'Type':<8} {'Precision':<12} {'Recall':<12} {'F-score':<12} {'Chamfer':<12}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        scene_name = row['scene']
        scene_type = row['scene_type']
        precision = row['precision']
        recall = row['recall']
        fscore = row['fscore']
        chamfer = row['chamfer']
        
        print(f"{scene_name:<15} {scene_type:<8} {precision:>11.4f} {recall:>11.4f} {fscore:>11.4f} {chamfer:>11.6f}")
    
    print("-" * 100)
    
    # 计算平均值
    print_color("\n整体平均:", Colors.GREEN)
    avg_metrics = df[['precision', 'recall', 'fscore', 'chamfer']].mean()
    print(f"  Precision@5cm: {avg_metrics['precision']:.4f} ({avg_metrics['precision']*100:.2f}%)")
    print(f"  Recall@5cm:    {avg_metrics['recall']:.4f} ({avg_metrics['recall']*100:.2f}%)")
    print(f"  F-score@5cm:   {avg_metrics['fscore']:.4f}")
    print(f"  Chamfer Dist:  {avg_metrics['chamfer']:.6f}")
    
    # 按场景类型统计
    print_color("\n按场景类型统计:", Colors.BLUE)
    for scene_type in ['normal', 'large', 'larger']:
        type_df = df[df['scene_type'] == scene_type]
        if len(type_df) > 0:
            type_avg = type_df[['precision', 'recall', 'fscore', 'chamfer']].mean()
            print(f"\n  {scene_type.upper()} scenes ({len(type_df)}):")
            print(f"    Precision: {type_avg['precision']:.4f}")
            print(f"    Recall:    {type_avg['recall']:.4f}")
            print(f"    F-score:   {type_avg['fscore']:.4f}")
            print(f"    Chamfer:   {type_avg['chamfer']:.6f}")
    
    print_color("\n" + "="*80, Colors.BLUE)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(exp_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存详细结果CSV
    csv_file = os.path.join(results_dir, f"scannetpp_evaluation_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    print_color(f"详细结果已保存到: {csv_file}", Colors.GREEN)
    
    # 保存总结JSON
    summary = {
        'timestamp': timestamp,
        'num_scenes': len(all_scenes),
        'num_evaluated': success_count,
        'overall_avg': {
            'precision': float(avg_metrics['precision']),
            'recall': float(avg_metrics['recall']),
            'fscore': float(avg_metrics['fscore']),
            'chamfer': float(avg_metrics['chamfer'])
        },
        'by_type': {}
    }
    
    for scene_type in ['normal', 'large', 'larger']:
        type_df = df[df['scene_type'] == scene_type]
        if len(type_df) > 0:
            type_avg = type_df[['precision', 'recall', 'fscore', 'chamfer']].mean()
            summary['by_type'][scene_type] = {
                'count': len(type_df),
                'precision': float(type_avg['precision']),
                'recall': float(type_avg['recall']),
                'fscore': float(type_avg['fscore']),
                'chamfer': float(type_avg['chamfer'])
            }
    
    json_file = os.path.join(results_dir, f"scannetpp_summary_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print_color(f"总结已保存到: {json_file}", Colors.GREEN)
    
    return True

if __name__ == '__main__':
    success = run_batch_evaluation()
    sys.exit(0 if success else 1)
