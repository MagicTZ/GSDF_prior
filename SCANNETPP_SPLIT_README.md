# ScanNet++ Train/Test Split 说明

## 📋 概述

为了与VoxelGS等方法进行公平对比，GSDF项目已经修改为支持使用ScanNet++官方的`train_test_lists.json`文件来划分训练集和测试集。

## 🔍 背景

### 为什么需要使用官方split？

1. **数据划分不同**：
   - **默认方式**：每8帧取1帧作为test（约12.5%数据）
   - **ScanNet++官方**：使用`train_test_lists.json`指定（约5%数据作为test）

2. **Mesh评估的特殊性**：
   - Mesh重建需要从多个视角融合深度信息
   - 使用更多训练视角（~316张）能获得更完整的几何重建
   - VoxelGS在评估时**只使用train集生成的mesh**

3. **公平对比原则**：
   - 所有方法应使用相同的train/test划分
   - Mesh评估应在train集上进行（更多视角，更完整重建）
   - NVS评估应在test集上进行（unseen views）

## 📁 文件格式

`train_test_lists.json` 示例：

```json
{
    "has_masks": false,
    "train": [
        "DSC06199.JPG",
        "DSC06200.JPG",
        ...
        "DSC06518.JPG"
    ],
    "test": [
        "DSC06520.JPG",
        "DSC06521.JPG",
        ...
        "DSC06536.JPG"
    ]
}
```

## 🔄 代码修改

### 修改位置

1. **SDF分支** (`instant_nsr/datasets/colmap.py`)
   - 在数据加载时检查是否存在`train_test_lists.json`
   - 如果存在，使用官方split
   - 否则回退到默认的每8帧策略

2. **GS分支** (`gaussian_splatting/scene/dataset_readers.py`)
   - 类似的逻辑
   - 按照`train_test_lists.json`划分train/test相机

### 关键逻辑

```python
# 检查是否存在官方split文件
train_test_file = os.path.join(root_dir, 'train_test_lists.json')
if os.path.exists(train_test_file):
    # 使用官方split
    with open(train_test_file, 'r') as f:
        split_data = json.load(f)
    
    train_images = set(split_data['train'])
    test_images = set(split_data['test'])
    
    # 根据图像名称划分
    if image_name in train_images:
        train_cam_infos.append(cam_info)
    elif image_name in test_images:
        test_cam_infos.append(cam_info)
else:
    # 回退到默认策略（每8帧）
    if idx % 8 != 0:
        train_cam_infos.append(cam_info)
    else:
        test_cam_infos.append(cam_info)
```

## 🚀 使用方法

### 1. 准备数据

确保ScanNet++数据目录包含`train_test_lists.json`：

```
Data/ScanNetpp/0a184cf634/
├── images/
│   ├── DSC06199.JPG
│   ├── DSC06200.JPG
│   └── ...
├── sparse/
│   └── 0/
├── train_test_lists.json  ← 必需文件
└── mesh.ply (GT mesh)
```

### 2. 训练

运行训练脚本（无需修改）：

```bash
bash train_scannetpp_smart.sh
```

程序会自动检测并使用`train_test_lists.json`。

**预期输出**：
```
Found train_test_lists.json, using official train/test split
  Train set: 316 images
  Test set: 17 images
Loaded 316 train cameras, 17 test cameras
```

### 3. 评估

- **NVS评估**（PSNR/SSIM/LPIPS）：在test集上评估
- **Mesh评估**（Chamfer Distance）：使用train集生成的mesh

## 📊 数据统计

以场景 `0a184cf634` 为例：

| Split | 图像数量 | 用途 |
|-------|---------|------|
| Train | 316 | Mesh重建 + 训练 |
| Test  | 17  | NVS评估 |

## ⚠️ 注意事项

1. **文件必需性**：
   - 对于ScanNet++数据集，必须包含`train_test_lists.json`
   - 如果缺失，程序会回退到默认split（但与VoxelGS不一致）

2. **Mesh评估协议**：
   - VoxelGS的`eval_recon.py`硬编码了使用`/train`路径的mesh
   - GSDF也应该导出train集的mesh用于评估

3. **坐标系对齐**：
   - 别忘了使用之前实现的坐标转换功能
   - 确保评估mesh在原始坐标系下

4. **图像名称匹配**：
   - 代码通过图像文件名（如`DSC06199.JPG`）匹配split
   - 确保`train_test_lists.json`中的名称与实际文件名一致

## 🔗 相关资源

- VoxelGS评估代码：`VoxelGS_dev/eval/eval_recon.py`
- ScanNet++官方文档：[https://kaldir.vc.in.tum.de/scannetpp/](https://kaldir.vc.in.tum.de/scannetpp/)

## 📝 验证方法

训练后检查日志：

```bash
# 检查是否使用了正确的split
grep "train_test_lists.json" exp/scene/with_prior@*/outputs.log

# 检查train/test相机数量
grep "Loaded.*cameras" exp/scene/with_prior@*/outputs.log
```

预期看到：
```
Found train_test_lists.json, using official train/test split
Loaded 316 train cameras, 17 test cameras
```

---

**更新日期**: 2026-01-25  
**适用版本**: GSDF v1.0+
