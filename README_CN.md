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
## PTQ QAT and deploy
若需要部署，请遵循 horizon 部署指南。

如果版本为 3、2 或 1，您可以尝试修改**view_transformerFisheye.py**中的代码：

`fused = bev_img_fp * bev_depth_fp` to `fused = bev_img_fp + bev_depth_fp`

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

- [https://github.com/ZouJiu1/bevPool](https://github.com/ZouJiu1/bevPool)
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

## 📐 附录：数学原理——矩阵等价变换 $PAH$（本项目最初的出发点，from myself）

整个项目源于我自己的一个最初想法：**图像特征图的每个通道是一个矩阵 $A \in \mathbb{R}^{50 \times 50}$，目标 BEV 特征图的每个通道也是一个矩阵 $Q \in \mathbb{R}^{128 \times 128}$——"矩阵变矩阵"最直接的线性代数工具就是左右各乘一个矩阵：$Q = PAH$**。左乘 $P$ 相当于行变换（作用在高度轴），右乘 $H$ 相当于列变换（作用在宽度轴）。因此视图变换器里图像变 BEV 只需要矩阵即可，不需要 LSS。

### $PAH$ 能把 $A$ 变成"任意"矩阵吗？两种情形的严格结论

**情形一：$P$、$H$ 可逆（初等行、列变换 / 相抵变换）**

- **等价标准形定理**：$\mathrm{rank}(A) = r$ 的 $m \times n$ 矩阵 $A$，存在可逆 $P$、$H$ 使 $PAH = E_r = \begin{pmatrix} I_r & O \\ O & O \end{pmatrix}$。
- **证明思路（高斯消元的矩阵语言）**：三类初等行/列变换各对应一个可逆的初等矩阵（$E(i,j)$ 换行、$E(i(c))$ 缩放、$E(i,j(k))$ 倍加），做一次行变换 ⟺ 左乘初等矩阵，做一次列变换 ⟺ 右乘初等矩阵。第一步行变换消成阶梯形 $P_s\cdots P_1 A = \begin{pmatrix} U_r \\ O \end{pmatrix}$；第二步列变换用主元清零右侧得 $\mathrm{diag}(D_r, O)$；第三步左乘 $\mathrm{diag}(d_1^{-1},\dots,d_r^{-1},1,\dots,1)$ 把主元缩放为 1。所有左乘攒成 $P$、右乘攒成 $H$，初等矩阵之积仍可逆。$\blacksquare$
- **相抵充要条件**：存在可逆 $P$、$H$ 使 $B = PAH$ ⟺ $\mathrm{rank}(A) = \mathrm{rank}(B)$。
  - ($\Rightarrow$) 可逆变换是线性同构，不改秩：$\mathrm{rank}(B) = \mathrm{rank}(PAH) = \mathrm{rank}(A)$；
  - ($\Leftarrow$) 设秩均为 $r$，则 $A = P_A^{-1}E_rH_A^{-1}$、$B = P_B^{-1}E_rH_B^{-1}$，消去 $E_r$ 得 $B = (P_B^{-1}P_A)\,A\,(H_AH_B^{-1})$。$\blacksquare$
- **可达集合：$\{Q : \mathrm{rank}(Q) = \mathrm{rank}(A)\}$，所有同型、同秩矩阵——不是全部矩阵。** 不变量：秩。

**情形二：$P$、$H$ 任意（不一定可逆）**

- **引理**：$\mathrm{rank}(XY) \le \min(\mathrm{rank}\,X, \mathrm{rank}\,Y)$（因为列空间 $\mathrm{Col}(XY) \subseteq \mathrm{Col}(X)$、行空间 $\mathrm{Row}(XY) \subseteq \mathrm{Row}(Y)$）。连续用两次得
$$\mathrm{rank}(PAH) \le \min(\mathrm{rank}\,P, \mathrm{rank}\,A, \mathrm{rank}\,H) \le \mathrm{rank}(A)$$
- **反向可达**：任意 $\mathrm{rank}(Q) = s \le r$ 的同型 $Q$ 均可被达到。构造：$Q = P_QE_sH_Q$，取对角投影阵 $D_s = \mathrm{diag}(1,\dots,1,0,\dots,0)$（前 $s$ 个为 1），由 $E_s = D_sE_rD_s'$ 与 $E_r = P_AAH_A$ 得 $Q = (P_QD_sP_A)\,A\,(H_AD_s'H_Q)$。$\blacksquare$
- **可达集合：$\{Q : \mathrm{rank}(Q) \le \mathrm{rank}(A)\}$。** 上限：结果矩阵秩不超过 $\mathrm{rank}(A)$。

**几何角度理解**：矩阵 $A$ 代表一个线性映射 $T: V \to W$。右乘可逆 $H$ ⟺ 换定义域 $V$ 的基（$x = Hx'$），左乘可逆 $P$ ⟺ 换值域 $W$ 的基（$y' = Py$）。换基不改变映射的秩——$\mathrm{rank}(A) = \dim\mathrm{Im}(T)$ 是映射的内在属性（像空间维数固定），与坐标系选取无关，所以换基之后映射的"有效维度"不变，自然无法得到秩更大的映射。

### 落到本项目

代码中 `matmul(img_feat, param)`（右乘，作用在宽度轴 $W$，列变换）与 permute 后的 `matmul`（等价左乘，作用在高度轴 $H$，行变换）合起来正是 $Q = PAH$。以 v2 为例：宽度链 $H = H_1H_2 \in \mathbb{R}^{50 \times 128}$，高度链 $P = P_2^{\top}P_1^{\top} \in \mathbb{R}^{128 \times 50}$。

**笔者的几何解释（逐元素看"特征移动与累加"）**：

- 右乘 $H$：$(AH)_{ij} = \sum_k A_{ik}H_{kj}$，对 $A$ 的**每一行**（同一高度上沿宽度排列的特征），$H$ 把第 $k$ 个宽度位置的特征以权重 $H_{kj}$ 移动并累加到新的第 $j$ 个位置；每行独立处理，行与行之间不混合；
- 左乘 $P$：$(PA)_{ij} = \sum_k P_{ik}A_{kj}$，对 $A$ 的**每一列**（同一宽度上沿高度排列的特征），$P$ 把第 $k$ 个高度位置的特征移动并累加到新的第 $i$ 个位置；每列独立处理，列与列之间不混合；
- 合起来：$Q_{ij} = \sum_{k,l} P_{ik}A_{kl}H_{lj}$——每个输入像素 $(k,l)$ 以可分离权重 $P_{ik}H_{lj}$ 贡献到每个输出像素 $(i,j)$，二维搬运被分解为"先沿宽度、再沿高度"两次一维搬运。

**通道共享**：投影矩阵形状为 `[num_views, ...]`，只按视角区分、不含通道维；`matmul` 对批次维 $B$ 与通道维 $C$ 广播，**同一图片的所有特征通道共用同一套 $P$、$H$**。空间搬运是通道无关的，通道混合由投影前的 1×1 卷积（`feat_net`/`depth_net`）负责——1×1 卷积只混通道、不移动空间位置；投影矩阵只移动空间位置、不混通道，两者互补。

秩分析：$A \in \mathbb{R}^{50 \times 50}$，故单通道 $\mathrm{rank}(Q) \le 50$（$Q$ 是 $128 \times 128$）——这是纯矩阵形式的固有天花板。缓解因素：

1. **64 个通道**：秩约束是每通道级别，64 通道各自承载不同低秩结构；
2. **4 路视角求和**：$\mathrm{rank}(B_1+B_2) \le \mathrm{rank}(B_1)+\mathrm{rank}(B_2)$，四个秩 $\le 50$ 的矩阵相加可铺满 128 维；
3. **BEV 编码器非线性**：打破线性形式的秩约束。

同理，v2/v3 的纯矩阵链在非线性缺位时可折叠为单个矩阵（$P_2^{\top}P_1^{\top}$ 仍是 $128 \times 50$），秩上限与 v1 相同，其收益来自优化动力学而非表达上限。

### 下一步计划（TODO）：突破 $\mathrm{rank}(A)$ 天花板的两个方案

**方案一：仿射变换 $Q = PAH + B$**

- **做法**：放弃纯乘法形式，加可学习偏移矩阵 $B \in \mathbb{R}^{128 \times 128}$，把线性映射升级为仿射映射（同构于 $y = Wx \to y = Wx + b$）。
- **为什么有效**：$B$ 可满秩 128，$\mathrm{rank}(PAH + B)$ 不再受 $\mathrm{rank}(A)$ 约束（极端例子：$P = H = O$、$B = I_{128}$ 时 $Q$ 满秩）；几何上引入"平移自由度"，特征零点可移动。
- **代价**：每视角每分支 $+16384$ 参数，4 视角 $\times$ 2 分支约 $0.13$M；加法是规则内存访问算子，部署友好性不变。
- **代码改动点**：`__init__` 增加 `self.bias_bev = nn.Parameter(torch.zeros(num_views, 1, grid_size[0], grid_size[1]))`，前向中 `flat_bev = flat_bev + self.bias_bev[i]`。
- **验证实验**：v1/v2/v3 分别加偏移，对比 NDS/mAP；可视化 $B$ 是否学到"先验占据图"结构。

**方案二：向量视角的线性组合 $Q = \sum_{k=1}^{K} P_k A H_k$**

- **做法**：把矩阵看成向量空间里的向量，用 $K$ 项混合投影替代单次左右乘。
- **数学依据（克罗内克积）**：$\mathrm{vec}(AXB) = (B^{\top} \otimes A)\,\mathrm{vec}(X)$，故 $\mathrm{vec}(PAH) = (H^{\top} \otimes P)\,\mathrm{vec}(A)$——$PAH$ 只是"克罗内克结构化"的线性映射，用 12800 参数表示 $\mathbb{R}^{2500} \to \mathbb{R}^{16384}$ 的映射（一般线性映射需约 4100 万参数）；$K$ 项求和 $\mathrm{vec}(Q) = \sum_k (H_k^{\top} \otimes P_k)\,\mathrm{vec}(A)$ 对应克罗内克秩 $\le K$ 的映射族，$K$ 足够大时可逼近任意线性映射。
- **为什么有效（秩视角）**：$\mathrm{rank}(Q) \le \sum_k \mathrm{rank}(P_kAH_k) \le K \cdot \mathrm{rank}(A) = 50K$；$K = 3$ 时上限 $150 \ge 128$，允许满秩。
- **与现有设计的联系**：4 路视角求和 = 跨视角 $K = 4$ 的雏形，双分支 = $K = 2$；本方案细化到单视角内部，每视角学 $K$ 套 $(P_k, H_k)$。
- **代价**：参数与计算随 $K$ 线性增长；用 `bmm`/`einsum` 把 $K$ 作批次维一次算完，仍是纯规则算子。
- **验证实验**：$K = 1, 2, 3, 4$ 消融观察 NDS/mAP 饱和点；与方案一组合 $Q = \sum_k P_kAH_k + B$。

**几何角度的统一理解**：纯 $PAH$ 的换基操作永远得不到秩更大的映射；方案一的 $+B$ 引入平移自由度（线性 → 仿射），方案二的求和引入多映射叠加自由度——两者从不同方向突破秩约束。

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
