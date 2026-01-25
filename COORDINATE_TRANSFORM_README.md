# 坐标系转换说明文档

## 📋 概述

GSDF项目在训练过程中会对场景进行归一化处理，包括：
1. **平移 (Shift)**: 将场景中心移动到原点
2. **缩放 (Scale)**: 对场景进行尺度归一化

这会导致生成的mesh与原始坐标系不一致。为了与Ground Truth mesh进行准确的评估对比，我们需要将归一化后的mesh转换回原始坐标系。

## 🔄 变换关系

### 前向变换（归一化）
```
vertices_normalized = (vertices_original - center) / scale
```

### 逆变换（还原到原始坐标系）
```
vertices_original = vertices_normalized * scale + center
```

## 📁 变换参数文件

训练时会自动生成变换参数JSON文件：

- **SDF分支**: `data/your_scene/transform_params_sdf.json`
- **GS分支**: `data/your_scene/transform_params_gs.json`

文件格式示例：
```json
{
  "center": [0.5, 0.3, 0.2],
  "scale": 2.5,
  "inv_trans": [[1, 0, 0, -0.5], [0, 1, 0, -0.3], [0, 0, 1, -0.2], [0, 0, 0, 1]]
}
```

## 🚀 使用方法

### 方法1: 训练时自动转换（推荐）

修改后的代码会在mesh导出时自动应用逆变换。导出的mesh文件名会包含 `_original` 后缀，表示已转换到原始坐标系。

**训练命令保持不变：**
```bash
bash train_scannetpp_smart.sh
```

**导出的mesh：**
- `it30000-mc1024.ply` - 归一化坐标系下的mesh
- `it30000-mc1024_original.ply` - **原始坐标系下的mesh** ✓

### 方法2: 使用转换工具手动转换

如果你已经有了归一化坐标系下的mesh，可以使用 `transform_mesh.py` 工具进行转换。

#### 转换单个mesh文件

```bash
python transform_mesh.py \
    --input exp/scene/trial/save/it30000-mc1024.ply \
    --transform data/scene/transform_params_sdf.json \
    --output mesh_original.ply
```

#### 批量转换目录下所有mesh

```bash
python transform_mesh.py \
    --input_dir exp/scene/trial/save/ \
    --transform data/scene/transform_params_sdf.json \
    --output_dir ./meshes_original/ \
    --suffix _original
```

#### 手动指定变换参数

如果没有保存的变换参数文件，可以手动指定：

```bash
python transform_mesh.py \
    --input mesh_normalized.ply \
    --manual \
    --center 0.5 0.3 0.2 \
    --scale 2.5 \
    --output mesh_original.ply
```

## 📊 与GT Mesh对比评估

转换后的mesh可以直接与原始坐标系下的GT mesh进行对比评估：

```bash
# 使用2DGS的评估工具计算Chamfer Distance
python eval_mesh.py \
    --pred meshes_original/it30000-mc1024_original.ply \
    --gt data/scene/gt_mesh.ply \
    --output metrics.json
```

## ⚙️ 技术细节

### 代码修改位置

1. **SDF分支数据加载** (`instant_nsr/datasets/colmap.py`):
   - 修改了 `simple_normalize_poses()` 函数，返回变换参数
   - 在数据加载时保存变换参数到JSON文件

2. **GS分支数据加载** (`gaussian_splatting/scene/dataset_readers.py`):
   - 在 `readColmapSceneInfo()` 函数中保存变换参数

3. **Mesh导出** (`instant_nsr/systems/neus.py`):
   - 修改了 `export()` 函数，在导出时自动应用逆变换

### 坐标系验证

导出mesh时会打印变换前后的bounding box信息：

```
Applied inverse transformation to mesh vertices
  Original bbox: [-3.1 -3.1 -3.1] to [3.1 3.1 3.1]
  Transformed bbox: [0.2 0.1 0.15] to [5.8 5.9 6.05]
```

可以通过对比这些数值与原始点云的范围来验证转换是否正确。

## 🐛 故障排除

### 问题1: 找不到变换参数文件

**症状**: `Warning: Transformation file transform_params_sdf.json not found`

**原因**: 使用旧代码训练的模型没有生成变换参数文件

**解决方案**: 
1. 重新运行数据预处理生成变换参数
2. 或使用工具脚本手动指定参数（需要从训练日志中查找）

### 问题2: 转换后mesh与GT不对齐

**检查清单**:
1. 确认使用了正确的变换参数文件（SDF分支用 `transform_params_sdf.json`）
2. 确认GT mesh确实在原始坐标系下
3. 检查归一化方式：自动计算 vs 手动指定（`neuralangelo_scale/center`）

### 问题3: Mesh出现在错误的位置

**原因**: 可能混用了不同来源的变换参数

**解决方案**: 
- SDF分支生成的mesh应使用 `transform_params_sdf.json`
- GS分支生成的mesh应使用 `transform_params_gs.json`（如果需要导出点云）

## 📝 注意事项

1. **训练时使用给定参数**: 如果在配置文件中指定了 `neuralangelo_scale` 和 `neuralangelo_center`，确保这些值被正确保存到变换参数文件中。

2. **Mesh截断**: 记住mesh在导出时会被截断到 `[-radius, radius]` 的范围内（通常 `radius=3.1`），超出部分会被丢弃。

3. **精度**: 变换参数以float32精度保存，通常足够准确。如需更高精度，可以修改保存格式。

## 🔗 参考

- 相关issue: 如何将mesh转换回原始坐标系
- 评估工具: [2DGS Mesh Evaluation](https://github.com/hbb1/2d-gaussian-splatting)
- GSDF论文: [arXiv:2403.16964](https://arxiv.org/abs/2403.16964)

---

有问题或建议？欢迎提issue！
