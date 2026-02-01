"""
Mesh extraction utilities for GSDF using TSDF fusion
Adapted from VoxelGS mesh extraction pipeline
"""

import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
import time


def to_cam_open3d(viewpoint_stack):
    """将GSDF的相机转换为Open3D格式"""
    camera_stack = []
    for viewpoint_cam in viewpoint_stack:
        # 获取相机参数
        W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
        fx = W / (2 * np.tan(viewpoint_cam.FovX / 2.))
        fy = H / (2 * np.tan(viewpoint_cam.FovY / 2.))
        cx, cy = W / 2., H / 2.
        
        # 内参矩阵
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(W, H, fx, fy, cx, cy)
        
        # 外参矩阵 (world_to_camera)
        extrinsic = np.zeros((4, 4))
        extrinsic[:3, :3] = viewpoint_cam.R.T
        extrinsic[:3, 3] = viewpoint_cam.T
        extrinsic[3, 3] = 1.0
        
        # 创建Open3D相机
        cam_o3d = o3d.camera.PinholeCameraParameters()
        cam_o3d.intrinsic = intrinsic
        cam_o3d.extrinsic = extrinsic
        
        camera_stack.append(cam_o3d)
    
    return camera_stack


def post_process_mesh(mesh, cluster_to_keep=100):
    """
    后处理mesh：保留最大的N个连通分量
    
    Args:
        mesh: Open3D mesh
        cluster_to_keep: 保留的最大连通分量数量
    
    Returns:
        cleaned mesh
    """
    print(f"Post-processing mesh: keeping top {cluster_to_keep} clusters...")
    
    # 去除重复和退化的三角形
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    
    # 聚类连通分量
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    
    if len(cluster_area) == 0:
        print("  Warning: No clusters found")
        return mesh
    
    # 按面积排序并保留最大的N个
    largest_cluster_idx = cluster_area.argsort()[::-1][:cluster_to_keep]
    triangles_to_remove = np.isin(triangle_clusters, largest_cluster_idx, invert=True)
    mesh.remove_triangles_by_mask(triangles_to_remove)
    
    print(f"  Kept {min(cluster_to_keep, len(cluster_area))} largest clusters")
    print(f"  Removed {triangles_to_remove.sum()} triangles")
    
    return mesh


class GaussianExtractor(object):
    def __init__(self, gaussians, render_func, prefilter_func, pipeline, bg_color=None):
        """
        从Gaussian Splatting提取mesh的工具类
        
        Args:
            gaussians: GaussianModel对象
            render_func: 渲染函数
            prefilter_func: 预过滤函数
            pipeline: PipelineParams
            bg_color: 背景颜色
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        
        self.gaussians = gaussians
        self.render_func = render_func
        self.prefilter_func = prefilter_func
        self.pipeline = pipeline
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        self.clean()
    
    def clean(self):
        """清空缓存"""
        self.rgbmaps = []
        self.depthmaps = []
        self.alphamaps = []
        self.viewpoint_stack = []
    
    @torch.no_grad()
    def reconstruction(self, viewpoint_stack, use_depth_filter=False):
        """
        重建场景：从所有视角渲染深度图
        
        Args:
            viewpoint_stack: 相机列表
            use_depth_filter: 是否使用深度过滤
        
        Returns:
            fps, visible_count, per_view_dict
        """
        print("\nRendering depth maps for TSDF fusion...")
        
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        
        t_list = []
        per_view_dict = {}
        
        for i, viewpoint_cam in tqdm(enumerate(viewpoint_stack), desc="Rendering progress"):
            # 渲染
            torch.cuda.synchronize()
            t0 = time.time()
            
            voxel_visible_mask = self.prefilter_func(
                viewpoint_cam, self.gaussians, self.pipeline, self.background
            )
            
            render_pkg = self.render_func(
                viewpoint_cam, self.gaussians, self.pipeline, self.background,
                visible_mask=voxel_visible_mask, out_depth=True, return_normal=True
            )
            
            torch.cuda.synchronize()
            t1 = time.time()
            t_list.append(t1 - t0)
            
            # 提取渲染结果
            rgb = render_pkg['render']
            depth = render_pkg['depth_hand']
            
            # Alpha从深度计算（简化版）
            alpha = (depth > 0).float()
            
            # 保存结果
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            self.alphamaps.append(alpha.cpu())
            
            per_view_dict[f'{i:05d}.png'] = depth.numel()
        
        # 堆叠所有图像
        self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        self.depthmaps = torch.stack(self.depthmaps, dim=0)
        self.alphamaps = torch.stack(self.alphamaps, dim=0)
        
        # 计算FPS
        fps = 1.0 / np.mean(t_list[5:]) if len(t_list) > 5 else 1.0 / np.mean(t_list)
        
        print(f"  Rendered {len(viewpoint_stack)} views")
        print(f"  Rendering FPS: {fps:.2f}")
        
        return fps, 0, per_view_dict
    
    def extract_mesh_bounded(self, voxel_size=0.015, sdf_trunc=0.075, 
                           depth_trunc=5.0, alpha_thres=0.5):
        """
        使用TSDF融合提取mesh（有界场景）
        
        Args:
            voxel_size: TSDF体素大小（米）
            sdf_trunc: SDF截断距离
            depth_trunc: 最大深度范围
            alpha_thres: Alpha阈值（过滤低置信度区域）
        
        Returns:
            Open3D mesh对象
        """
        print("\n" + "=" * 60)
        print("Running TSDF volume integration...")
        print(f"  voxel_size: {voxel_size}m")
        print(f"  sdf_trunc: {sdf_trunc}m")
        print(f"  depth_trunc: {depth_trunc}m")
        print(f"  alpha_threshold: {alpha_thres}")
        print("=" * 60)
        
        # 创建TSDF volume
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        # 转换相机格式
        cameras_o3d = to_cam_open3d(self.viewpoint_stack)
        
        # 逐帧融合
        for i, cam_o3d in tqdm(enumerate(cameras_o3d), 
                               desc="TSDF integration", 
                               total=len(cameras_o3d)):
            
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            alpha = self.alphamaps[i]
            
            # 过滤低置信度区域
            depth = depth.clone()
            depth[alpha < alpha_thres] = 0
            
            # 创建Open3D RGBD图像
            rgb_o3d = o3d.geometry.Image(
                np.asarray(
                    np.clip(rgb.permute(1, 2, 0).numpy(), 0.0, 1.0) * 255,
                    order="C",
                    dtype=np.uint8
                )
            )
            
            depth_o3d = o3d.geometry.Image(
                np.asarray(
                    depth.squeeze(0).numpy(),
                    order="C",
                    dtype=np.float32
                )
            )
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d,
                depth_o3d,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0
            )
            
            # 融合到volume中
            volume.integrate(rgbd, cam_o3d.intrinsic, cam_o3d.extrinsic)
        
        # 提取mesh
        print("\nExtracting mesh from TSDF volume...")
        mesh = volume.extract_triangle_mesh()
        
        print(f"  Extracted mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
        
        return mesh


def export_mesh_with_tsdf(gaussians, render_func, prefilter_func, pipeline, 
                          viewpoint_stack, output_path, bg_color=[0, 0, 0],
                          voxel_size=0.015, depth_trunc=5.0, 
                          num_clusters=100, use_post_process=True):
    """
    完整的mesh导出流程：渲染 → TSDF融合 → 后处理
    
    Args:
        gaussians: GaussianModel
        render_func: render函数
        prefilter_func: prefilter_voxel函数
        pipeline: PipelineParams
        viewpoint_stack: 相机列表
        output_path: 输出路径（不含扩展名）
        bg_color: 背景颜色
        voxel_size: TSDF体素大小
        depth_trunc: 深度截断
        num_clusters: 保留的最大连通分量数
        use_post_process: 是否使用后处理
    
    Returns:
        mesh, mesh_post (后处理的mesh)
    """
    # 创建提取器
    extractor = GaussianExtractor(
        gaussians, render_func, prefilter_func, pipeline, bg_color
    )
    
    # 1. 渲染深度图
    fps, _, _ = extractor.reconstruction(viewpoint_stack, use_depth_filter=False)
    
    # 2. TSDF融合提取mesh
    sdf_trunc = 5.0 * voxel_size
    mesh = extractor.extract_mesh_bounded(
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc,
        alpha_thres=0.5
    )
    
    # 3. 保存原始mesh
    mesh_path = output_path + "_tsdf.ply"
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"\n✓ Saved raw TSDF mesh: {mesh_path}")
    
    # 4. 后处理
    if use_post_process:
        print("\nPost-processing mesh...")
        mesh_post = post_process_mesh(mesh, cluster_to_keep=num_clusters)
        
        # 保存后处理的mesh
        mesh_post_path = output_path + "_tsdf_post.ply"
        o3d.io.write_triangle_mesh(mesh_post_path, mesh_post)
        print(f"✓ Saved post-processed mesh: {mesh_post_path}")
        
        return mesh, mesh_post
    
    return mesh, mesh
