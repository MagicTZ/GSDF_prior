#!/usr/bin/env python3
"""
GSDF Mesh Evaluation Script
借鉴VoxelGS的评估方式，评估转换到原始坐标系的mesh与GT mesh
"""

import os
import json
import numpy as np
import open3d as o3d
from argparse import ArgumentParser
from pathlib import Path

def read_point_cloud(path_cloud):
    """读取点云/mesh文件"""
    assert os.path.exists(path_cloud), f"File not found: {path_cloud}"
    cloud = o3d.io.read_point_cloud(path_cloud)
    return cloud

def nn_correspondance(verts1, verts2):
    """计算最近邻对应关系"""
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances

def visualize_error_map(pcd_pred, dist2, errormap_path, error_bound=0.10):
    """可视化误差图"""
    from matplotlib import cm
    print(f'Visualizing error map...')

    dist_score = dist2.clip(0, error_bound) / error_bound
    color_map = cm.get_cmap('Reds')
    colors = color_map(dist_score)[:, :3]
    pcd_pred.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(errormap_path, pcd_pred)
    print(f'Error map saved to {errormap_path}')

def evaluate_gsdf_mesh(pred_mesh_path, gt_mesh_path, 
                       threshold=0.05, down_sample=0.02, 
                       show_errormap=True, output_dir=None):
    """
    评估GSDF生成的mesh
    
    Args:
        pred_mesh_path: 预测mesh路径 (应该是转换到原始坐标系的_original.ply)
        gt_mesh_path: GT mesh路径
        threshold: 距离阈值 (m)
        down_sample: 体素下采样大小 (m)
        show_errormap: 是否生成误差可视化
        output_dir: 输出目录
    
    Returns:
        Dict of metrics
    """
    print("=" * 60)
    print("GSDF Mesh Evaluation")
    print("=" * 60)
    print(f"Pred mesh: {pred_mesh_path}")
    print(f"GT mesh:   {gt_mesh_path}")
    print(f"Threshold: {threshold}m")
    print(f"Down sample: {down_sample}m")
    
    # 检查文件存在性
    if not os.path.exists(pred_mesh_path):
        print(f"Error: Prediction mesh not found at {pred_mesh_path}")
        return None
    
    if not os.path.exists(gt_mesh_path):
        print(f"Error: GT mesh not found at {gt_mesh_path}")
        return None
    
    # 读取mesh
    print("\nLoading meshes...")
    pcd_pred = read_point_cloud(pred_mesh_path)
    pcd_trgt = read_point_cloud(gt_mesh_path)
    
    print(f"  Pred vertices: {len(pcd_pred.points)}")
    print(f"  GT vertices:   {len(pcd_trgt.points)}")
    
    # 下采样
    if down_sample:
        print(f"\nDownsampling with voxel size {down_sample}m...")
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
        print(f"  Pred vertices after downsample: {len(pcd_pred.points)}")
        print(f"  GT vertices after downsample:   {len(pcd_trgt.points)}")
    
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)
    
    # 计算对应关系和距离
    print("\nComputing correspondences...")
    ind1, dist1 = nn_correspondance(verts_pred, verts_trgt)  # gt->pred
    ind2, dist2 = nn_correspondance(verts_trgt, verts_pred)  # pred->gt
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)
    
    # 计算指标
    print("\nComputing metrics...")
    precision = np.mean((dist2 < threshold).astype('float'))
    recall = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    chamfer = np.mean(dist1**2) + np.mean(dist2**2)
    
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'fscore': float(fscore),
        'chamfer': float(chamfer),
        'dist_pred_to_gt': float(np.mean(dist2)),  # pred->gt 平均距离
        'dist_gt_to_pred': float(np.mean(dist1)),  # gt->pred 平均距离
    }
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"  Precision@{threshold}m: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall@{threshold}m:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F-score@{threshold}m:   {fscore:.4f}")
    print(f"  Chamfer Distance: {chamfer:.6f}")
    print(f"  Pred->GT mean dist: {np.mean(dist2):.4f}m")
    print(f"  GT->Pred mean dist: {np.mean(dist1):.4f}m")
    print("=" * 60)
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 'mesh_evaluation_results.json')
        with open(result_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {result_file}")
    
    # 生成误差可视化
    if show_errormap:
        if output_dir is None:
            output_dir = os.path.dirname(pred_mesh_path)
        errormap_path = os.path.join(output_dir, 'mesh_error_map.ply')
        visualize_error_map(pcd_pred, dist2, errormap_path, error_bound=threshold*2)
    
    return metrics


def main():
    parser = ArgumentParser(description="Evaluate GSDF mesh against GT")
    parser.add_argument('--pred_mesh', '-p', type=str, required=True,
                       help='Path to predicted mesh (should be *_original.ply in original coordinate)')
    parser.add_argument('--gt_mesh', '-g', type=str, required=True,
                       help='Path to ground truth mesh')
    parser.add_argument('--threshold', '-t', type=float, default=0.05,
                       help='Distance threshold for precision/recall (default: 0.05m = 5cm)')
    parser.add_argument('--down_sample', '-d', type=float, default=0.02,
                       help='Voxel size for downsampling (default: 0.02m = 2cm)')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='Output directory for results and error map')
    parser.add_argument('--no_errormap', action='store_true',
                       help='Disable error map visualization')
    
    args = parser.parse_args()
    
    # 运行评估
    metrics = evaluate_gsdf_mesh(
        pred_mesh_path=args.pred_mesh,
        gt_mesh_path=args.gt_mesh,
        threshold=args.threshold,
        down_sample=args.down_sample,
        show_errormap=not args.no_errormap,
        output_dir=args.output_dir
    )
    
    if metrics is None:
        print("\nEvaluation failed!")
        return 1
    
    print("\n✓ Evaluation completed successfully!")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

'''
# 基本用法
python eval_gsdf_mesh.py \
    --pred_mesh exp/0a184cf634/with_prior@*/save/it30000-mc1024_original.ply \
    --gt_mesh /root/autodl-tmp/Proj/GS-Reconstruction/Data/ScanNetpp/0a184cf634/mesh.ply \
    --output_dir exp/0a184cf634/with_prior@*/evaluation

# 自定义参数
python eval_gsdf_mesh.py \
    -p exp/0a184cf634/with_prior@*/save/it30000-mc1024_original.ply \
    -g data/0a184cf634/mesh.ply \
    -t 0.05 \
    -d 0.02 \
    -o results/
'''