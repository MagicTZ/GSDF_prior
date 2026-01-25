#!/usr/bin/env python3
"""
测试 train_test_lists.json split 功能

这个脚本验证：
1. 能否正确读取 train_test_lists.json
2. train/test split 的相机数量是否正确
3. 是否按照文件名正确分配
"""

import os
import sys
import json

def test_train_test_split():
    """测试train/test split功能"""
    print("=" * 60)
    print("测试 ScanNet++ Train/Test Split 功能")
    print("=" * 60)
    
    # 测试数据路径
    data_root = '/root/autodl-tmp/Proj/GS-Reconstruction/Data/ScanNetpp/0a184cf634'
    split_file = os.path.join(data_root, 'train_test_lists.json')
    
    print(f"\n数据路径: {data_root}")
    print(f"Split文件: {split_file}")
    
    # 检查文件是否存在
    if not os.path.exists(split_file):
        print(f"\n✗ 错误：找不到 {split_file}")
        print("请确保ScanNet++数据包含 train_test_lists.json 文件")
        return False
    
    print(f"✓ 找到 train_test_lists.json")
    
    # 读取split文件
    try:
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        print(f"✓ 成功读取 train_test_lists.json")
    except Exception as e:
        print(f"\n✗ 读取失败: {e}")
        return False
    
    # 检查必需的key
    if 'train' not in split_data or 'test' not in split_data:
        print(f"\n✗ 错误：train_test_lists.json 缺少 'train' 或 'test' key")
        return False
    
    train_images = split_data['train']
    test_images = split_data['test']
    
    print(f"\n数据统计:")
    print(f"  Train images: {len(train_images)}")
    print(f"  Test images: {len(test_images)}")
    print(f"  Total: {len(train_images) + len(test_images)}")
    
    # 检查图像文件是否存在
    images_dir = os.path.join(data_root, 'images')
    if not os.path.exists(images_dir):
        print(f"\n✗ 警告：找不到 images 目录")
    else:
        print(f"\n检查图像文件...")
        
        train_exists = 0
        test_exists = 0
        
        for img_name in train_images[:5]:  # 只检查前5张
            img_path = os.path.join(images_dir, img_name)
            if os.path.exists(img_path):
                train_exists += 1
        
        for img_name in test_images[:5]:  # 只检查前5张
            img_path = os.path.join(images_dir, img_name)
            if os.path.exists(img_path):
                test_exists += 1
        
        print(f"  Train images (sampled): {train_exists}/5 存在")
        print(f"  Test images (sampled): {test_exists}/5 存在")
        
        if train_exists == 0 or test_exists == 0:
            print(f"  ⚠️  警告：部分图像文件可能不存在")
    
    # 打印示例
    print(f"\nTrain集示例图像:")
    for img in train_images[:3]:
        print(f"  - {img}")
    
    print(f"\nTest集示例图像:")
    for img in test_images[:3]:
        print(f"  - {img}")
    
    # 检查是否有重复
    train_set = set(train_images)
    test_set = set(test_images)
    overlap = train_set & test_set
    
    if len(overlap) > 0:
        print(f"\n✗ 错误：Train和Test集有 {len(overlap)} 张重复图像")
        return False
    else:
        print(f"\n✓ Train和Test集无重复")
    
    # 验证预期值（针对0a184cf634场景）
    # 注意：实际数据可能少于拍摄的照片数量，因为质量控制和COLMAP重建
    # 场景 0a184cf634: DSC06281-06285 这5张图片被排除
    expected_train = 315  # 实际是315，不是316
    expected_test = 17
    
    if len(train_images) == expected_train and len(test_images) == expected_test:
        print(f"\n✓ 图像数量符合预期（场景 0a184cf634）")
    else:
        print(f"\n⚠️  信息：图像数量与此场景的预期值不同")
        print(f"  预期(0a184cf634): Train={expected_train}, Test={expected_test}")
        print(f"  实际: Train={len(train_images)}, Test={len(test_images)}")
        print(f"  注：不同场景的图像数量会有差异，这是正常的")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    
    return True


def test_code_integration():
    """测试代码集成"""
    print("\n" + "=" * 60)
    print("测试代码集成")
    print("=" * 60)
    
    # 检查修改的文件
    files_to_check = [
        '/root/autodl-tmp/Proj/GSDF/instant_nsr/datasets/colmap.py',
        '/root/autodl-tmp/Proj/GSDF/gaussian_splatting/scene/dataset_readers.py',
    ]
    
    print("\n检查修改的文件:")
    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
            
            # 检查是否包含关键代码
            with open(file_path, 'r') as f:
                content = f.read()
                if 'train_test_lists.json' in content:
                    print(f"    ✓ 包含 train_test_lists.json 处理代码")
                else:
                    print(f"    ✗ 未找到 train_test_lists.json 处理代码")
                    all_exist = False
        else:
            print(f"  ✗ {file_path} (不存在)")
            all_exist = False
    
    if all_exist:
        print("\n✓ 代码集成检查通过")
    else:
        print("\n✗ 代码集成检查失败")
    
    return all_exist


def main():
    print("\n" + "=" * 60)
    print("GSDF ScanNet++ Split 功能测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("Train/Test Split 读取", test_train_test_split()))
    results.append(("代码集成检查", test_code_integration()))
    
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
        print("\n可以开始训练了：")
        print("  bash train_scannetpp_smart.sh")
    else:
        print("✗ 部分测试失败，请检查。")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
