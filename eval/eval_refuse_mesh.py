#!/usr/bin/env python3
"""
TSDF Refusion & Evaluation Script for GSDF Project
功能：
1. GT 裁剪 (Clipping): 根据 GT BBox 裁剪无关区域
2. TSDF 重融合 (Refusion): 提高 Mesh 表面质量 (使用 pyrender 渲染深度)
3. 评估 (Evaluation): 计算 Acc, Comp, Precision, Recall, F-score
"""
import sys
import os

# 设置 pyrender 使用 EGL 后端（headless 渲染）
os.environ['PYOPENGL_PLATFORM'] = 'OSMesa' # egl

# 添加GSDF项目路径
GSDF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GSDF_ROOT)

import numpy as np
import open3d as o3d
import trimesh
import pyrender
from tqdm import tqdm
from pathlib import Path
import argparse
import json
from sklearn.neighbors import KDTree


def load_colmap_poses_and_intrinsics(data_root):
    """从COLMAP数据中加载相机位姿和内参"""
    from instant_nsr.datasets.colmap_utils import \
        read_cameras_binary, read_images_binary, read_cameras_text, read_images_text
    
    # 读取相机内参
    cam_path = os.path.join(data_root, 'sparse/0/cameras.bin')
    if os.path.exists(cam_path):
        camdata = read_cameras_binary(cam_path)
    else:
        cam_path = os.path.join(data_root, 'colmap/cameras.txt')
        camdata = read_cameras_text(cam_path)
    
    H = int(camdata[1].height)
    W = int(camdata[1].width)
    
    # 解析相机内参
    if camdata[1].model == 'SIMPLE_RADIAL' or camdata[1].model == 'SIMPLE_PINHOLE':
        fx = fy = camdata[1].params[0]
        cx = camdata[1].params[1]
        cy = camdata[1].params[2]
    elif camdata[1].model in ['PINHOLE', 'OPENCV']:
        fx = camdata[1].params[0]
        fy = camdata[1].params[1]
        cx = camdata[1].params[2]
        cy = camdata[1].params[3]
    else:
        raise ValueError(f"Unsupported camera model: {camdata[1].model}")
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    # 读取相机位姿
    img_path = os.path.join(data_root, 'sparse/0/images.bin')
    if os.path.exists(img_path):
        imdata = read_images_binary(img_path)
    else:
        img_path = os.path.join(data_root, 'colmap/images.txt')
        imdata = read_images_text(img_path)
    
    poses = []
    sorted_keys = sorted(imdata.keys())
    
    # 每隔N张采样
    sample_interval = 10
    for i, key in enumerate(sorted_keys[::sample_interval]):
        d = imdata[key]
        R = d.qvec2rotmat()
        t = d.tvec.reshape(3, 1)
        # 转换为c2w
        c2w = np.concatenate([R.T, -R.T @ t], axis=1)
        c2w = np.vstack([c2w, [0, 0, 0, 1]])
        poses.append(c2w)
    
    poses = np.array(poses)
    
    return poses, K, (W, H)


# ==========================================
# Pyrender-based Renderer (更稳定)
# ==========================================

class PyrenderDepthRenderer:
    """使用 pyrender 进行深度渲染的类"""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
    
    def render_depth(self, mesh_pyrender, pose, intrinsics):
        """
        渲染深度图
        Args:
            mesh_pyrender: pyrender.Mesh 对象
            pose: 4x4 camera-to-world 矩阵
            intrinsics: 3x3 或 4x4 内参矩阵
        Returns:
            depth: (H, W) 深度图
        """
        # 更新视口大小
        self.renderer.viewport_width = self.width
        self.renderer.viewport_height = self.height
        
        # 清空场景并添加 mesh
        self.scene.clear()
        self.scene.add(mesh_pyrender)
        
        # 提取内参
        if intrinsics.shape[0] == 4:
            K = intrinsics[:3, :3]
        else:
            K = intrinsics
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # 创建 pyrender 相机
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        
        # pyrender 使用 OpenGL 坐标系，需要对 pose 进行变换
        # OpenGL: Y up, -Z forward
        # OpenCV/Colmap: Y down, +Z forward
        # 需要绕 X 轴旋转 180 度
        pose_opengl = self._fix_pose_for_opengl(pose)
        
        self.scene.add(camera, pose=pose_opengl)
        
        # 渲染
        _, depth = self.renderer.render(self.scene)
        
        return depth
    
    def _fix_pose_for_opengl(self, pose):
        """将 OpenCV/Colmap 坐标系的 pose 转换为 OpenGL 坐标系"""
        # 绕 X 轴旋转 180 度
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R_flip = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R_flip
        return pose @ axis_transform
    
    def delete(self):
        """释放渲染器资源"""
        self.renderer.delete()


def tsdf_refusion_pyrender(mesh, poses, K, img_wh, voxel_size=0.01, depth_trunc=5.0):
    """
    使用 pyrender 渲染深度图，然后进行 TSDF 重融合
    这个方法比 Open3D 的 RaycastingScene 更稳定
    
    Args:
        mesh: trimesh.Trimesh 对象
        poses: (N, 4, 4) camera-to-world 矩阵
        K: (3, 3) 内参矩阵
        img_wh: (W, H) 图像尺寸
        voxel_size: TSDF 体素大小
        depth_trunc: 深度截断距离
    
    Returns:
        refined_mesh: trimesh.Trimesh 对象
    """
    W, H = img_wh
    
    print(f"[Refusion] 初始化 pyrender 渲染器 ({W}x{H})...")
    renderer = PyrenderDepthRenderer(width=W, height=H)
    
    # 将 trimesh 转换为 pyrender mesh
    print(f"[Refusion] 转换 mesh 到 pyrender 格式...")
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    
    # 创建 TSDF 体积
    print(f"[Refusion] 创建 TSDF 体积 (voxel_size={voxel_size}m)...")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=3 * voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    # 构建 4x4 内参矩阵
    intrinsic_4x4 = np.eye(4)
    intrinsic_4x4[:3, :3] = K
    
    print(f"[Refusion] 开始 TSDF 融合，共 {len(poses)} 个视角...")
    success_count = 0
    
    for idx, pose in enumerate(tqdm(poses, desc="TSDF融合")):
        try:
            # 使用 pyrender 渲染深度
            depth = renderer.render_depth(mesh_pyrender, pose, intrinsic_4x4)
            
            # 过滤无效深度
            depth = depth.astype(np.float32)
            depth[depth <= 0] = 0
            depth[depth > depth_trunc] = 0
            
            # 创建 RGB 图像（白色占位）
            rgb = np.ones((H, W, 3), dtype=np.uint8) * 255
            
            # 转换为 Open3D 图像
            rgb_o3d = o3d.geometry.Image(rgb)
            depth_o3d = o3d.geometry.Image(depth)
            
            # 创建 RGBD 图像
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d,
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False
            )
            
            # 构建 Open3D 相机内参
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
                width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy
            )
            
            # 计算 world-to-camera 外参
            extrinsic = np.linalg.inv(pose)
            
            # 集成到 TSDF 体积
            volume.integrate(rgbd, intrinsic_o3d, extrinsic)
            success_count += 1
            
        except Exception as e:
            print(f"\n[Warning] 视角 {idx+1}/{len(poses)} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n[Refusion] 成功融合 {success_count}/{len(poses)} 个视角")
    
    # 释放渲染器
    renderer.delete()
    
    # 提取 mesh
    print("[Refusion] 从 TSDF 体积提取 mesh...")
    refined_mesh_o3d = volume.extract_triangle_mesh()
    
    # 转换为 trimesh
    refined_mesh = trimesh.Trimesh(
        vertices=np.asarray(refined_mesh_o3d.vertices),
        faces=np.asarray(refined_mesh_o3d.triangles),
        process=False
    )
    
    print(f"[Refusion] 完成! 顶点数: {len(refined_mesh.vertices)}, 面数: {len(refined_mesh.faces)}")
    
    return refined_mesh


def sample_points(mesh, num_samples=200000):
    """从 Mesh 表面均匀采样点云"""
    if isinstance(mesh, trimesh.Trimesh):
        points, _ = trimesh.sample.sample_surface(mesh, num_samples)
        return points
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
        return np.asarray(pcd.points)
    return np.array([])


def nn_correspondance(verts1, verts2):
    """计算 verts1 到 verts2 的最近邻距离"""
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts2)
    distances, indices = kdtree.query(verts1, k=1)
    distances = distances.reshape(-1)
    return distances


def compute_metrics(mesh_pred, mesh_gt, threshold=0.05, num_samples=200000):
    """计算 Acc, Comp, Prec, Recall, F-score"""
    print(f"[Metrics] Sampling {num_samples} points from meshes...")
    verts_pred = sample_points(mesh_pred, num_samples)
    verts_gt = sample_points(mesh_gt, num_samples)
    
    print(f"[Metrics] Calculating distances...")
    # dist1: Pred -> GT (Accuracy / Precision)
    dist1 = nn_correspondance(verts_pred, verts_gt)
    # dist2: GT -> Pred (Completeness / Recall)
    dist2 = nn_correspondance(verts_gt, verts_pred)
    
    chamfer = np.mean(dist1**2) + np.mean(dist2**2)

    accuracy = np.mean(dist1)
    completeness = np.mean(dist2)
    
    precision = np.mean((dist1 < threshold).astype('float'))
    recall = np.mean((dist2 < threshold).astype('float'))
    fscore = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        'Accuracy': accuracy,
        'Completeness': completeness,
        'Precision': precision,
        'Recall': recall,
        'F-score': fscore,
        'Threshold': threshold,
        'Chamfer': chamfer
    }


def clip_mesh_by_gt(pred_mesh, gt_mesh, margin=0.05):
    """根据 GT BBox 裁剪 Pred Mesh"""
    print(f"[Clip] Clipping mesh with margin {margin}m...")
    bbox_min = gt_mesh.bounds[0] - margin
    bbox_max = gt_mesh.bounds[1] + margin
    
    print(f"  GT BBox Min: {bbox_min}")
    print(f"  GT BBox Max: {bbox_max}")
    
    # 找出在 BBox 内的顶点
    mask = np.all((pred_mesh.vertices >= bbox_min) & (pred_mesh.vertices <= bbox_max), axis=1)
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) == 0:
        print("[Error] No vertices left after clipping!")
        return pred_mesh
    
    # 保留面
    face_mask = np.all(np.isin(pred_mesh.faces, valid_indices), axis=1)
    
    if not np.any(face_mask):
         print("[Error] No complete faces left after clipping!")
         return pred_mesh

    clipped_mesh = pred_mesh.copy()
    clipped_mesh.update_faces(face_mask)
    clipped_mesh.remove_unreferenced_vertices()
    
    print(f"  Vertices: {len(pred_mesh.vertices)} -> {len(clipped_mesh.vertices)}")
    return clipped_mesh


def main():
    parser = argparse.ArgumentParser(description='TSDF Refusion & Evaluation Tool')
    
    # I/O Parameters
    parser.add_argument('--mesh', type=str, required=True, help='输入mesh文件路径 (.ply)')
    parser.add_argument('--data_root', type=str, default=None, help='COLMAP数据根目录 (Refusion需提供)')
    parser.add_argument('--gt', type=str, default=None, help='GT mesh路径 (裁剪/评估需提供)')
    parser.add_argument('--output', type=str, default=None, help='输出mesh文件路径')
    
    # Action Switches
    parser.add_argument('--skip_refuse', action='store_true', help='跳过 Refusion 步骤')
    parser.add_argument('--use_clip', action='store_true', help='使用 GT BBox 裁剪 (需提供 --gt)')
    
    # Refusion Parameters
    parser.add_argument('--voxel_size', type=float, default=0.01, help='TSDF体素大小')
    parser.add_argument('--depth_trunc', type=float, default=5.0, help='深度截断距离')
    parser.add_argument('--sample_interval', type=int, default=10, help='相机采样间隔')
    parser.add_argument('--downsample', type=int, default=2, help='图像下采样倍数')
    parser.add_argument('--simplify_faces', type=int, default=500000, help='简化mesh到指定面数用于渲染 (未使用)')
    
    # Evaluation/Clipping Parameters
    parser.add_argument('--clip_margin', type=float, default=0.05, help='裁剪Margin (米)')
    parser.add_argument('--eval_threshold', type=float, default=0.05, help='F-score 阈值 (米)')
    parser.add_argument('--eval_samples', type=int, default=200000, help='评估采样点数')

    args = parser.parse_args()
    
    # 检查参数依赖
    if not args.skip_refuse and args.data_root is None:
        print("[Error] Performing Refusion requires --data_root. Use --skip_refuse to skip.")
        sys.exit(1)
    if args.use_clip and args.gt is None:
        print("[Error] Clipping requires --gt.")
        sys.exit(1)
    
    # 设置输出路径
    if args.output is None:
        mesh_path = Path(args.mesh)
        suffix = ""
        if args.use_clip: suffix += "_clipped"
        if not args.skip_refuse: suffix += "_tsdf"
        if suffix == "": suffix = "_processed"
        args.output = str(mesh_path.parent / f"{mesh_path.stem}{suffix}{mesh_path.suffix}")
    
    # 1. 加载初始 Mesh
    print(f"Loading Mesh: {args.mesh}")
    current_mesh = trimesh.load(args.mesh, process=False)
    print(f"  Vertices: {len(current_mesh.vertices)}, Faces: {len(current_mesh.faces)}")
    
    stats = {
        'original_vertices': len(current_mesh.vertices),
        'original_faces': len(current_mesh.faces)
    }

    # 2. Clipping (Step 1)
    # 逻辑调整：先 Clip 去掉无用区域，再进行 Refusion，可以加速并提高 Refusion 稳定性
    if args.use_clip and args.gt:
        print(f"\n[Step 1] Clipping by GT BBox...")
        print(f"Loading GT: {args.gt}")
        gt_mesh = trimesh.load(args.gt, process=False)
        
        current_mesh = clip_mesh_by_gt(current_mesh, gt_mesh, margin=args.clip_margin)
        
        stats['clipped_vertices'] = len(current_mesh.vertices)
        stats['clipped_faces'] = len(current_mesh.faces)
        
        # 总是保存裁剪后的中间结果，便于 Debug
        clipped_output = str(Path(args.output).parent / f"{Path(args.mesh).stem}_clipped.ply")
        print(f"Saving Clipped Mesh to {clipped_output}")
        current_mesh.export(clipped_output)

        # 如果跳过 refusion，那么最终输出就是这个 clipped mesh
        if args.skip_refuse:
             print(f"Copying Clipped Mesh to Final Output: {args.output}")
             current_mesh.export(args.output)
        
    elif not args.use_clip:
        print("\n[Step 1] Skipping Clipping (not enabled)")

    # 3. Refusion (Step 2) - 使用 pyrender
    if not args.skip_refuse:
        print(f"\n[Step 2] Refusion processing (using pyrender)...")
        print(f"Loading camera parameters: {args.data_root}")
        poses, K, img_wh = load_colmap_poses_and_intrinsics(args.data_root)
        
        # 下采样
        W, H = img_wh
        W = W // args.downsample
        H = H // args.downsample
        K = K / args.downsample
        K[2, 2] = 1.0
        img_wh = (W, H)
        
        print(f"  Views: {len(poses)}, Resolution: {W}x{H}")
        print(f"  Voxel Size: {args.voxel_size}m")
        
        current_mesh = tsdf_refusion_pyrender(
            current_mesh, poses, K, img_wh,
            voxel_size=args.voxel_size,
            depth_trunc=args.depth_trunc
        )
        
        stats['tsdf_vertices'] = len(current_mesh.vertices)
        stats['tsdf_faces'] = len(current_mesh.faces)
        
        print(f"Saving Refused Mesh to {args.output}")
        current_mesh.export(args.output)
        
    elif args.skip_refuse and not args.use_clip:
        print("\n[Info] No processing steps (refuse/clip) selected.")
        
    # 4. Evaluation (Step 3)
    if args.gt:
        print(f"\n[Step 3] Evaluation...")
        if 'gt_mesh' not in locals():
            gt_mesh = trimesh.load(args.gt, process=False)
            
        metrics = compute_metrics(
            current_mesh, gt_mesh, 
            threshold=args.eval_threshold, 
            num_samples=args.eval_samples
        )
        
        print("\n" + "="*40)
        print(" FINAL RESULTS ")
        print("="*40)
        for k, v in metrics.items():
            if 'Threshold' in k: continue
            print(f"{k:15s}: {v:.5f}")
        print("="*40 + "\n")
        
        stats.update(metrics)
        
    # 保存统计信息
    stats_file = Path(args.output).parent / f"{Path(args.output).stem}_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {stats_file}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
