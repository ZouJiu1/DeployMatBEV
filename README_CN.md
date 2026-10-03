# DeployMatBEV

面向 **NPU 部署** 的多鱼眼 BEV 3D 目标检测——自学习矩阵视图变换器。

> 🔗 [English README](README.md)

> PDF [DeployMatBEV PDF](paper_outputs/paper_main.pdf)
---

## 概述

环视鱼眼相机为自动驾驶和自动泊车提供了低成本 360° 感知能力。然而，主流 BEV 检测器依赖 `grid_sample`、2D `gather`、`scatterND`、one-hot 等算子——这些算子的**不规则、数据依赖的内存访问模式**与嵌入式 NPU 架构存在根本性矛盾。

**DeployMatBEV** 将显式几何驱动的视图变换整体替换为**可学习的分离矩阵投影**。核心数据路径仅包含：
- 矩阵乘法（`matmul`）
- 张量置换（`permute`）
- 通道维度 softmax
- 逐元素乘法与加法

所有算子均具有**规则、连续的内存访问模式**——这正是 NPU 硬件设计所加速的操作类型。

---

## 🔥 核心亮点

| 特性 | DeployMatBEV | Fisheye3DOD（基线） |
|:-----|:------------:|:-------------------:|
| **NDS** | **0.4801** | 0.4853 |
| **mASE** | **0.1319** | 0.1637 |
| **mAOE** | **0.3826** | 0.4799 |
| `grid_sample` | ❌ 无 | ✅ 使用 |
| 2D `gather` | ❌ 无 | ✅ 使用 |
| `scatterND` / BEVPool | ❌ 无 | ✅ 使用 |
| 推理需相机外参 | ❌ 不需要 | ✅ 需要 |
| 总参数量 | **14.5M**（v3） | 19.5M |
| 量化友好 | ✅ PTQ / QAT | 部分支持 |

- 与几何基线的 NDS 差距仅 **0.0052**，同时参数量减少 **25.8%**
- 推理时**无需相机外参**（`lidar2camera`、`ego2camera`）
- 三个容量版本（`learning_point_version = 1/2/3`），共享同一套 NPU 友好算子
- 在 **nuScenes 针孔**（6 相机）上初步验证，证实算子集可跨相机模型迁移

---

## 🏗️ 网络架构

```
4× 鱼眼图片输入 (800×800)
        ↓
   EfficientNet-B0 图像编码器
        ↓
  FastSCNN 风格特征 Neck（64 通道）
        ↓        ↓
   图像特征分支    深度门控分支（1×1 卷积 + 通道 softmax）
        ↓                  ↓
   自学习分离矩阵投影（H×W → H_b×W_b）
        ↓                  ↓
     B_i^F  ⊙  B_i^D   （逐元素融合）
        ↓
   4 路求和 → 统一 BEV（128×128）
        ↓
   BEV 编码器 + Center-based 解码器
        ↓
   3D 框 [x, y, z, w, l, h, yaw]
```

核心算子为 [`model/view_transformerFisheye.py`](model/view_transformerFisheye.py) 中的 `_spatial_transfom_learn` 函数，沿高度和宽度维度学习视图特定的投影矩阵——无需参考点、无需 BEV 网格、无需采样。

---

## 🗂️ 模型库

### Fisheye3DOD（4 路环视鱼眼相机）

Fisheye3DOD pretrained models download link: [https://www.alipan.com/s/HKq1YjCYVjj 提取码: 78yp](https://www.alipan.com/s/HKq1YjCYVjj)

| Version | `learning_point_version` | Checkpoint | NDS | mAP | mATE | mASE | mAOE | Params |
|:-------:|:------------------------:|:-----------|:---:|:---:|:----:|:----:|:----:|:------:|
| v1 | `1` | [`float-checkpoint-best1.pth.tar`](https://www.alipan.com/s/HKq1YjCYVjj) | 0.4361 | 0.3047 | 0.7160 | 0.1324 | 0.4489 | 12.8M |
| v2 | `2` | [`float-checkpoint-best.pth2.tar`](https://www.alipan.com/s/HKq1YjCYVjj) | 0.4392 | 0.3132 | 0.7202 | 0.1399 | 0.4448 | 13.4M |
| v3 | `3` | [`float-checkpoint-best3.pth.tar`](https://www.alipan.com/s/HKq1YjCYVjj) | 0.4766 | 0.3533 | 0.6478 | **0.1319** | 0.4209 | 14.5M |
| v3* | `3` | [`float-checkpoint-best333.pth.tar`](https://www.alipan.com/s/HKq1YjCYVjj) | **0.4801** | 0.3515 | 0.6578 | 0.1333 | **0.3826** | 14.5M |
> **v3*** 表示延长训练、方向估计改进后的版本。

### 基线

| 方法 | 权重文件 | NDS | mAP | 参数量 |
|:-----|:--------|:---:|:---:|:------:|
| Fisheye3DOD | [`fisheye_bevdet.pth`](model/ckpt/fisheye_bevdet.pth) | 0.4853 | 0.3821 | 19.5M |

> 原始 Fisheye3DOD 采用 ResNet-18 + FPN 图像骨干网络、SECOND BEV 骨干网络，额外存储 **1.44M 不可学习几何 buffer**（frustum 和球面网格）。

### nuScenes（6 路针孔相机，初步结果）

| 变体 | `learning_point_version` | NDS | mAP |
|:-----|:------------------------:|:---:|:---:|
| Extended v9 | `9` | 0.1717 | 0.0599 |
| Extended v10 | `10` | 0.1895 | 0.0788 |
| Extended v11 | `11` | **0.1903** | **0.0773** |

> 这些为早期训练 checkpoint，仅作为跨相机模型的**通用性验证**，不代表与 nuScenes SOTA 检测器竞争的精度。

---

## 🛠️ 运行环境

使用地平线 OpenExplorer Docker 容器进行训练和评测。

**Docker 镜像**：`docker_open_explorer_ubuntu_22_j6_gpu_v3.8.1`

下载地址：[https://oe.horizon.auto/download/oe](https://oe.horizon.auto/download/oe)

---

## 📦 数据准备

### Fisheye3DOD

数据集来源：[https://github.com/weiyangdaren/Fisheye3DOD](https://github.com/weiyangdaren/Fisheye3DOD)

**第一步**：生成 info pickle（与官方仓库相同）：

```bash
python3 projects/Fisheye3DOD/tools/fisheye3dod_converter.py
```

**第二步**：打包为 LMDB 格式以高效加载：

```bash
python3 lmdbdata/fisheye_carla_packer.py \
  --src-data-dir ./Fisheye3DODdataset \
  --meta_json_dir ./ImageSets-2hz \
  --pack-type lmdb \
  --target-data-dir ./lmdb \
  --split-name train \
  --num-workers 10

python3 lmdbdata/fisheye_carla_packer.py \
  --src-data-dir ./Fisheye3DODdataset \
  --meta_json_dir ./ImageSets-2hz \
  --pack-type lmdb \
  --target-data-dir ./lmdb \
  --split-name val \
  --num-workers 10
```

**第三步**: 可视化LMDB内的数据

```bash
python3 imageCheck.py --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py --savePath ./output
```

### nuScenes

数据集来源：[https://github.com/nutonomy/nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

```bash
python3 lmdbdata/nuscenes_packer.py \
  --src-data-dir ./nuscenes \
  --version ./v1.0-trainval \
  --pack-type lmdb \
  --target-data-dir ./nuscenes_lmdb \
  --split-name train \
  --num-workers 10

python3 lmdbdata/nuscenes_packer.py \
  --src-data-dir ./nuscenes \
  --version ./v1.0-trainval \
  --pack-type lmdb \
  --target-data-dir ./nuscenes_lmdb \
  --split-name val \
  --num-workers 10
```

> `v1.0-mini` 仅用于探索，不用于训练。

---

## 🚀 训练

### 单卡

```bash
# Fisheye3DOD
python3 train.py --stage float \
  --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \
  --device-ids 0

# nuScenes
python3 train.py --stage float \
  --config ./config/nuscene_config.py \
  --device-ids 0
```

### 多卡

```bash
python3 train.py --stage float \
  --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \
  --device-ids 0,1
```

### 关键超参数

| 参数 | 值 |
|:-----|:--:|
| 输入分辨率 | 800 × 800 |
| 图像特征图尺寸 | 50 × 50（16× 下采样） |
| BEV 网格 | 128 × 128 |
| BEV 范围 | 51.2 m × 51.2 m（@ 0.4 m/cell） |
| 特征通道数 | 64 |
| 深度通道数 | 64 |
| 隐藏投影维度 R | 256 |
| 优化器 | AdamW（lr=2e-4，weight_decay=0.01） |
| 学习率调度 | Cyclic LR |
| 训练轮数 | 36（float） |
| GPU 数 | 4 |
| 每卡 batch | 1（有效 batch = 4） |

> 容量由配置文件中的 `learning_point_version` 控制：`1`（直接投影）、`2`（一个隐藏层，默认）、`3`（两个隐藏层）。

---

## 📊 评测

### Fisheye3DOD

```bash
# 版本 3（最佳 NDS）
python3 predict.py --stage float \
  --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \
  --device-ids 0 \
  --learning_point_version 3 \
  --ckpt ./model/ckpt/float-checkpoint-best3.pth.tar

# 版本 3*（延长训练）
python3 predict.py --stage float \
  --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \
  --device-ids 0 \
  --learning_point_version 3 \
  --ckpt ./model/ckpt/float-checkpoint-best333.pth.tar

# 版本 2
python3 predict.py --stage float \
  --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \
  --device-ids 0 \
  --learning_point_version 2 \
  --ckpt ./model/ckpt/float-checkpoint-best.pth2.tar

# 版本 1
python3 predict.py --stage float \
  --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \
  --device-ids 0 \
  --learning_point_version 1 \
  --ckpt ./model/ckpt/float-checkpoint-best1.pth.tar
```

### nuScenes

nuScenes pretrained models download link: [https://www.alipan.com/s/kAsevEPwv51 提取码: e81v](https://www.alipan.com/s/kAsevEPwv51)

```bash
python3 predict.py --stage float \
  --config ./config/nuscene_config.py \
  --device-ids 0 \
  --learning_point_version 11 \
  --ckpt ./model/ckptnuscene/float-checkpoint-best.pth111111.tar
```

---

## 🎨 可视化

Fisheye3DOD
```bash
# Version 3 (best NDS)
python3 infer_float.py \
  --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \
  --save-path ./infer_out \
  --model-inputs ./demo/fisheye3dod_demo/16 \
  --learning_point_version 3 \
  --ckpt ./model/ckpt/float-checkpoint-best3.pth.tar

# Version 3* (extended training)
python3 infer_float.py \
  --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \
  --save-path ./infer_out \
  --model-inputs ./demo/fisheye3dod_demo/16 \
  --learning_point_version 3 \
  --ckpt ./model/ckpt/float-checkpoint-best333.pth.tar

# Version 2
python3 infer_float.py \
  --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \
  --save-path ./infer_out \
  --model-inputs ./demo/fisheye3dod_demo/16 \
  --learning_point_version 2 \
  --ckpt ./model/ckpt/float-checkpoint-best.pth2.tar
```

nuScenes
```bash
python3 infer_float.py --config ./config/nuscene_config.py \
--save-path ./infer_out --model-inputs ./demo/bev_lss_efficientnetb0_multitask_nuscenes \
--learning_point_version 10 --ckpt ./model/ckptnuscene/float-checkpoint-best.pth101010.tar


python3 infer_float.py --config ./config/nuscene_config.py \
--save-path ./infer_out --model-inputs ./demo/bev_lss_efficientnetb0_multitask_nuscenes \
--learning_point_version 9 --ckpt ./model/ckptnuscene/float-checkpoint-best.pth99.tar

python3 infer_float.py --config ./config/nuscene_config.py \
--save-path ./infer_out --model-inputs ./demo/bev_lss_efficientnetb0_multitask_nuscenes \
--learning_point_version 11 --ckpt ./model/ckptnuscene/float-checkpoint-best.pth111111.tar
```

输出示例：

<img src="./infer_out/nusc_pred_10.jpg" width="100%"/>
<img src="./infer_out/nusc_pred_16.jpg" width="100%"/>

---

## 📤 ONNX 导出

Fisheye3DOD
```bash
python3 export_onnx.py \
  --config ./config/bev_lss_efficientnetb0_multitask_nuscenes.py \
  --learning_point_version 3 \
  --ckpt ./model/ckpt/float-checkpoint-best3.pth.tar
```

nuScenes
```bash
python3 export_onnx.py --config ./config/nuscene_config.py \
--learning_point_version 10 --ckpt ./model/ckptnuscene/float-checkpoint-best.pth101010.tar
```
---

## 📐 参数量统计

运行参数量统计脚本：

```bash
python3 model/statistic.py
```

| 模型 | 视图变换参数（可学习） | 总参数量 | Checkpoint 大小 |
|:-----|:---------------------:|:--------:|:---------------:|
| 原始 Fisheye3DOD | 0.05M + 1.44M buffer | 19.48M | 235.4 MB |
| DeployMatBEV v1 | 0.10M | 12.79M | 82.1 MB |
| DeployMatBEV v2 | 0.73M | 13.41M | 89.6 MB |
| DeployMatBEV v3 | 1.78M | 14.46M | 102.2 MB |

> Checkpoint 文件包含优化器状态；纯模型权重大约为一半。

## Reference

- [Fisheye3DOD](https://github.com/weiyangdaren/Fisheye3DOD)
- [horizon PTQ/QAT deployment guide](https://doc.oe.horizon.auto/3.8.1/guide/model_compile.html)
- [horizon developer portal](https://developer.horizon.auto/)
- [ocamcalib_undistort](https://github.com/matsuren/ocamcalib_undistort)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [lift-splat-shoot](https://github.com/nv-tlabs/lift-splat-shoot)
- [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)
- [Trae](https://www.trae.cn/)

---

## 📝 引用

如果本工作对您有帮助，请引用：

```bibtex
@article{zou2025deploymatbev,
  title={DeployMatBEV: A Self-Learned Matrix View Transformer for NPU-Deployable Multi-Fisheye BEV 3D Object Detection},
  author={Zou, Jiu},
  journal={github repository},
  year={2026}
}
```

---

## 📄 许可证

本项目基于 GPL-3.0 license 发布。

---

**联系方式**：Jiu Zou — [1069679911@qq.com](mailto:1069679911@qq.com)

Goyu Meta Technology Co., Ltd.

---

## 📎 附录：`learning_point_version` 设计思路 from myself

所有版本共享同一个核心思想：用**可学习的分离矩阵**沿图像高度（H）和宽度（W）轴进行投影，替代显式几何投影。各版本的差异在于投影深度、通道拆分、归一化方式以及深度分支是否启用。代码参考：[`_spatial_transfom_learn`](model/view_transformerFisheye.py#L1605-L2018)。

### Fisheye3DOD 版本（v1 / v2 / v3）

| | v1 | v2 | v3 |
|:--|:--|:--|:--|
| 每轴 matmul 次数 | 1 | 2 | 3 |
| 隐藏投影层数 | 0 | 1 | 2 |
| 深度分支 | ✅ 启用 | ✅ 启用 | ✅ 启用 |
| 融合方式 | `img_BEV × depth_BEV`（逐元素） | 相同 | 相同 |
| 多视角聚合 | 各视角求和 | 相同 | 相同 |

**设计逻辑：**
- **v1** —— 最简形式。每轴一次 matmul：`W_img(50→128)`，permute 后再 `H_img(50→128)`。验证纯学习矩阵即可完成视图变换。
- **v2** —— 每轴增加一个隐藏层（如 `50→256→128`），在算子集不变的前提下提升映射表达能力。
- **v3** —— 每轴两个隐藏层（`50→256→256→128`）。最深变体，精度最佳（NDS 0.4766–0.4801）。逐轴分解使计算量远低于稠密的 `(H·W)→(H_b·W_b)` 投影。
- 三者均采用**双分支设计**：图像特征与深度门控特征分别由独立矩阵组（`param_point*` / `dparam_point*`）投影，再逐元素融合。深度监督通过深度分支注入，起到软性遮挡/可见性门控的作用。

### nuScenes 版本（v9 / v10 / v11）

| | v9 | v10 | v11 |
|:--|:--|:--|:--|
| 基础结构 | v3 式，每轴 3 层 | v3 式，每轴 3 层 | v10 + 末尾输出投影 |
| 通道处理 | 拆为 3 × 64 通道组分别投影 | 全通道 | 全通道 |
| 归一化 | matmul 之间插入 BatchNorm | matmul 之间插入 BatchNorm | matmul 之间插入 BatchNorm |
| 残差连接 | ✅（每轴阶段的输入加回） | ✅ | ✅ |
| 深度分支 | ❌ 关闭（仅图像） | ❌ 关闭 | ❌ 关闭 |
| 融合方式 | 分组拼接 → BN → 视角拼接 → BN | 视角拼接 → BN | 视角拼接 → BN |

**设计逻辑：**
- **v9**（"SplitMatrix"）—— 将 192 通道拆为三个 64 通道组，每组独立投影后再合并。动机：让不同通道组专门学习不同的空间映射（分组投影），代价是参数量增加。
- **v10** —— 全通道 v3 式投影，在链式 matmul 之间插入 **BatchNorm + 残差连接**。这打破了纯 matmul 链的线性坍缩问题，并在更大的 nuScenes 数据集上稳定训练。为降低 6 相机 batch 的显存占用，关闭了深度分支。
- **v11** —— 在 v10 基础上，于第二轴阶段之后增加一个额外的最终投影矩阵，在输出 BEV 前多一步可学习的重映射。NDS/mAP 略优于 v10。

### 可优化与改进方向 from trae Kimi-K3

1. **为 v1–v3 引入非线性。** 纯链式 matmul 无非线性时在数学上可坍缩为单个矩阵。加入 BN/ReLU（如 v9–v11 所做）或低秩瓶颈，才能让 v3 的深度带来真正的表达力提升，而非仅优化动力学上的收益。
2. **在 v9–v11 中恢复深度门控。** 深度分支目前被注释掉；恢复 v1–v3 使用的 `img × depth` 门控，可能找回在 Fisheye3DOD 上观察到的方向/尺度收益。
3. **跨视角权重共享。** 目前每个视角拥有独立矩阵。共享（或部分共享，如低秩 + 每视角残差）可削减参数量，并在相机布局对称时提升泛化。
4. **批量化视角处理。** 逐视角的 Python 循环可替换为单个批量 matmul（`bmm`）或块对角形式，提高 GPU/NPU 利用率。
5. **几何初始化。** 用已知相机几何（如双线性采样权重）初始化学习矩阵而非随机初始化，可加速收敛并缩小与几何基线的 NDS 差距。
6. **投影链的量化感知。** `bev_quant`/`depth_bev_quant` QAT 钩子已在融合点存在；将 QAT 扩展到完整投影链可验证真实 NPU 硬件上的 INT8 精度损失。
7. **几何教师蒸馏。** 用显式几何基线作为学习矩阵的教师，是结合几何归纳偏置与 NPU 友好算子的低成本途径。
