# GSDF ScanNet++ 批量训练和评估指南

本文档介绍如何使用批量脚本进行多场景训练和评估。

## 场景列表

根据VoxelGS的划分标准，共11个场景:

### Normal Scenes (7个)
- 0a184cf634, 13c3e046d7, 1d003b07bd, 260db9cf5a, 8be0cd3817, 6464461276, 8b5caf3398

### Large Scenes (4个)
- 578511c8a9, 036bce3393, 6f1848d1e3, 281ba69af1

### Larger Scenes (1个)
- 9460c8889d

## 批量训练

### 方法1: Shell脚本并行训练 (推荐)

使用5个GPU (GPU 0-4) 真正并行训练:

```bash
cd /root/autodl-tmp/Proj/GSDF
chmod +x batch_train_parallel.sh
./batch_train_parallel.sh
```

特点:
- 真正的并行执行，5个GPU同时工作
- 每个场景的日志独立保存到 `exp/[scene_name]/training.log`
- 自动检测checkpoint并恢复训练
- 支持后台运行

查看特定场景的训练日志:
```bash
tail -f exp/0a184cf634/training.log
```

### 方法2: Python脚本顺序训练

```bash
cd /root/autodl-tmp/Proj/GSDF
python batch_train_scannetpp.py
```

特点:
- 顺序执行，每次使用一个GPU
- 更容易监控和调试
- 自动生成训练总结 `exp/training_summary.json`

## 批量评估

训练完成后，批量评估所有场景的mesh质量:

```bash
cd /root/autodl-tmp/Proj/GSDF
python batch_eval_scannetpp.py
```

### 评估标准

参考VoxelGS的评估标准:
- **Threshold**: 5cm (0.05m) for Precision/Recall/F-score
- **Downsample**: 2cm (0.02m) voxel downsampling
- **Metrics**:
  - Precision@5cm: 预测点到GT的准确率
  - Recall@5cm: GT点到预测的召回率
  - F-score@5cm: 精确率和召回率的调和平均
  - Chamfer Distance: 双向平均距离

### 输出结果

评估完成后会生成:

1. **详细CSV报告**: `exp/evaluation_results/scannetpp_evaluation_[timestamp].csv`
   - 每个场景的详细指标

2. **总结JSON**: `exp/evaluation_results/scannetpp_summary_[timestamp].json`
   - 整体平均指标
   - 按场景类型分组统计

3. **误差可视化**: 每个场景的 `mesh_error_map.ply`
   - 位于各场景的save目录下
   - 用红色深浅表示误差大小

### 查看结果示例

```bash
# 查看CSV结果
cat exp/evaluation_results/scannetpp_evaluation_*.csv

# 查看JSON总结
cat exp/evaluation_results/scannetpp_summary_*.json | jq .

# 用Open3D查看误差图
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('exp/0a184cf634/*/save/mesh_error_map.ply')])"
```

## 单个场景训练和评估

如果只想训练/评估单个场景:

### 训练单个场景

```bash
# 使用智能脚本 (自动检测checkpoint)
./train_scannetpp_smart.sh

# 或手动指定场景
exp_dir=./exp
config=configs/scannetpp/13c3e046d7.yaml
gpu=0
tag=with_prior

python launch.py \
    --exp_dir ${exp_dir} \
    --config ${config} \
    --gpu ${gpu} \
    --train \
    --eval \
    tag=${tag}
```

### 评估单个场景

```bash
python eval/eval_gsdf_mesh.py \
    --pred_mesh exp/0a184cf634/with_prior@*/save/it30000-mc1024_original.ply \
    --gt_mesh /root/autodl-tmp/Proj/GS-Reconstruction/Data/ScanNetpp/0a184cf634/mesh.ply \
    --threshold 0.05 \
    --down_sample 0.02 \
    --output_dir exp/0a184cf634/evaluation
```

## GPU使用说明

当前配置使用5个GPU (0-4)，如需修改:

### 修改可用GPU数量

编辑 `batch_train_parallel.sh`:
```bash
# 修改这一行，例如使用3个GPU
gpus=(0 1 2)
```

编辑 `batch_train_scannetpp.py`:
```python
# 修改这一行
available_gpus = [0, 1, 2]
```

## 监控训练进度

### 方法1: 查看日志文件

```bash
# 实时查看特定场景
tail -f exp/0a184cf634/training.log

# 查看所有场景的最新日志
tail -n 20 exp/*/training.log
```

### 方法2: 检查checkpoint

```bash
# 查看已完成的checkpoint
find exp/ -name "*.ckpt" -type f -exec ls -lh {} \;

# 按时间排序
find exp/ -name "*.ckpt" -type f -printf "%T@ %p\n" | sort -n
```

### 方法3: GPU使用情况

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或使用gpustat (如果安装)
watch -n 1 gpustat
```

## 故障排查

### 训练中断后恢复

脚本会自动检测最新的checkpoint并恢复训练，无需手动操作。

### 某个场景训练失败

查看对应的日志文件找出原因:
```bash
cat exp/[scene_name]/training.log
```

常见问题:
- OOM (显存不足): 减少batch size或使用更大的GPU
- 配置文件不存在: 确保 `configs/scannetpp/[scene_name].yaml` 存在
- 数据路径错误: 检查yaml中的 `root_dir` 是否正确

### 重新训练特定场景

删除该场景的checkpoint后重新运行:
```bash
rm -rf exp/0a184cf634
# 然后重新运行训练脚本
```

## 与VoxelGS对比

使用相同的评估标准，可以直接对比GSDF和VoxelGS的结果:

| Method | Precision@5cm | Recall@5cm | F-score | Chamfer |
|--------|---------------|------------|---------|---------|
| VoxelGS | (从VoxelGS结果) | | | |
| GSDF | (从本脚本输出) | | | |

## 依赖项

确保已安装:
```bash
pip install pandas numpy open3d matplotlib
```

## 预期训练时间

基于单个场景约8-12小时 (30K iterations):
- 使用5个GPU并行: 约20-30小时完成全部11个场景
- 顺序执行: 约90-130小时

## 目录结构

```
Proj/GSDF/
├── configs/scannetpp/          # 所有场景的配置文件
│   ├── 0a184cf634.yaml
│   ├── 13c3e046d7.yaml
│   └── ...
├── exp/                        # 实验输出
│   ├── 0a184cf634/
│   │   ├── with_prior@[timestamp]/
│   │   │   ├── save/           # mesh输出
│   │   │   ├── checkpoints/    # 训练checkpoint
│   │   │   └── csv_logs/
│   │   └── training.log        # 训练日志
│   ├── evaluation_results/     # 评估结果汇总
│   └── training_summary.json   # 训练总结
├── batch_train_parallel.sh     # 并行训练脚本
├── batch_train_scannetpp.py    # Python训练脚本
├── batch_eval_scannetpp.py     # 批量评估脚本
└── README_BATCH.md            # 本文档
```
