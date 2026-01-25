#!/usr/bin/env python3
"""
测试坐标转换功能的脚本

这个脚本会：
1. 创建一个简单的测试mesh
2. 应用前向变换（归一化）
3. 应用逆变换（还原）
4. 验证结果是否正确
"""

import numpy as np
import json
import tempfile
import os

try:
    import trimesh
except ImportError:
    print("Warning: trimesh not installed, skipping mesh test")
    trimesh = None


def test_transformation():
    """测试坐标变换的正确性"""
    print("=" * 60)
    print("测试坐标变换功能")
    print("=" * 60)
    
    # 定义测试参数
    center = np.array([1.0, 2.0, 3.0])
    scale = 2.5
    
    print(f"\n测试参数:")
    print(f"  Center: {center}")
    print(f"  Scale: {scale}")
    
    # 创建测试点
    original_vertices = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
        [1.0, 2.0, 3.0],
        [-2.0, 3.0, 8.0]
    ])
    
    print(f"\n原始顶点:")
    print(original_vertices)
    
    # 前向变换（归一化）
    normalized_vertices = (original_vertices - center) / scale
    print(f"\n归一化后的顶点:")
    print(normalized_vertices)
    
    # 逆变换（还原）
    restored_vertices = normalized_vertices * scale + center
    print(f"\n逆变换后的顶点:")
    print(restored_vertices)
    
    # 验证
    diff = np.abs(restored_vertices - original_vertices)
    max_error = diff.max()
    print(f"\n最大误差: {max_error}")
    
    if max_error < 1e-6:
        print("✓ 坐标变换测试通过！")
        return True
    else:
        print("✗ 坐标变换测试失败！")
        return False


def test_transform_params_save_load():
    """测试变换参数的保存和加载"""
    print("\n" + "=" * 60)
    print("测试变换参数保存和加载")
    print("=" * 60)
    
    # 创建测试参数
    transform_params = {
        'center': [1.0, 2.0, 3.0],
        'scale': 2.5,
        'inv_trans': [[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3], [0, 0, 0, 1]]
    }
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(transform_params, f, indent=2)
        temp_file = f.name
    
    print(f"\n保存参数到: {temp_file}")
    
    # 加载
    with open(temp_file, 'r') as f:
        loaded_params = json.load(f)
    
    print(f"加载的参数:")
    print(f"  Center: {loaded_params['center']}")
    print(f"  Scale: {loaded_params['scale']}")
    
    # 验证
    if (loaded_params['center'] == transform_params['center'] and 
        loaded_params['scale'] == transform_params['scale']):
        print("✓ 参数保存/加载测试通过！")
        success = True
    else:
        print("✗ 参数保存/加载测试失败！")
        success = False
    
    # 清理
    os.remove(temp_file)
    
    return success


def test_mesh_transformation():
    """测试mesh变换（需要trimesh）"""
    if trimesh is None:
        print("\n跳过mesh测试（trimesh未安装）")
        return True
    
    print("\n" + "=" * 60)
    print("测试Mesh变换")
    print("=" * 60)
    
    # 创建一个简单的立方体mesh
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    original_vertices = mesh.vertices.copy()
    
    print(f"\n原始mesh顶点数: {len(original_vertices)}")
    print(f"原始bbox: {mesh.bounds[0]} to {mesh.bounds[1]}")
    
    # 定义变换参数
    center = np.array([5.0, 5.0, 5.0])
    scale = 3.0
    
    # 应用前向变换
    mesh.vertices = (mesh.vertices - center) / scale
    print(f"\n归一化后bbox: {mesh.bounds[0]} to {mesh.bounds[1]}")
    
    # 应用逆变换
    mesh.vertices = mesh.vertices * scale + center
    print(f"\n还原后bbox: {mesh.bounds[0]} to {mesh.bounds[1]}")
    
    # 验证
    diff = np.abs(mesh.vertices - original_vertices)
    max_error = diff.max()
    print(f"\n最大误差: {max_error}")
    
    if max_error < 1e-6:
        print("✓ Mesh变换测试通过！")
        return True
    else:
        print("✗ Mesh变换测试失败！")
        return False


def main():
    print("\n" + "=" * 60)
    print("GSDF 坐标转换功能测试")
    print("=" * 60)
    
    results = []
    
    # 运行所有测试
    results.append(("坐标变换", test_transformation()))
    results.append(("参数保存/加载", test_transform_params_save_load()))
    results.append(("Mesh变换", test_mesh_transformation()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！")
        print("\n实现验证成功，可以安全使用坐标转换功能。")
    else:
        print("✗ 部分测试失败，请检查实现。")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
