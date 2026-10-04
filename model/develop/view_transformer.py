# Copyright (c) Horizon Robotics. All rights reserved.
###########################
# matmul version
import os
import collections
import logging
from collections.abc import Sequence
from typing import Any, Dict, List, Tuple

import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor, nn
from torch.overrides import handle_torch_function, has_torch_function
from hat.core.nus_box3d_utils import adjust_coords, get_min_max_coords
from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap
from ocamcamera import OcamCamera
import math

try:
    from hbdk.torch_script.placeholder import placeholder
except ImportError:
    placeholder = None

logger = logging.getLogger(__name__)

__all__ = ["LSSTransformerFisheye"]

class FixedGridSampleNearestConv(nn.Module):
    def __init__(self, grid_size, C, ):
        super().__init__()
        self.Hout, self.Wout = grid_size
        self.C = C

    def build_conv_weight(self, grid_fixed, Hin, Win):
        x_pix = 0.5 * (grid_fixed[..., 0] + 1) * (Win - 1)
        y_pix = 0.5 * (grid_fixed[..., 1] + 1) * (Hin - 1)
        x_idx = torch.round(x_pix).clamp(0, Win-1).long()
        y_idx = torch.round(y_pix).clamp(0, Hin-1).long()

        # 构造稀疏采样权重 [Hout*Wout, Hin*Win]
        weight = torch.zeros(self.Hout*self.Wout, Hin*Win, device=x_idx.device)
        oh_ow = 0
        for oh in range(self.Hout):
            for ow in range(self.Wout):
                y = y_idx[0, oh, ow].item()
                x = x_idx[0, oh, ow].item()
                pos = y * Win + x
                weight[oh_ow, pos] = 1.0
                oh_ow += 1
        # 分组卷积权重变形 [HoutWout, 1, Hin, Win]
        weight = weight.view(self.Hout*self.Wout, 1, Hin, Win)
        # 分组卷积，C组，每组单独卷积
        self.sampler = nn.Conv2d(
            self.C, self.C * self.Hout * self.Wout,
            kernel_size=(Hin, Win), stride=1, padding=0, groups=self.C,
            bias=False
        )
        with torch.no_grad():
            w_rep = weight.repeat(self.C, 1, 1, 1)
            self.sampler.weight.copy_(w_rep)

    def forward(self, feat):
        B, C, Hin, Win = feat.shape
        # Conv输出 [B, C*HoutWout, 1, 1]
        feat_conv = self.sampler(feat)
        # reshape还原 [B, C, Hout, Wout]
        out = feat_conv.view(B, C, self.Hout, self.Wout)
        return out

class GridSampleNearestReplace(nn.Module):
    # https://docs.pytorch.org/docs/2.12/generated/torch.gather.html
    def __init__(self, grid_size):
        super(GridSampleNearestReplace, self).__init__()
        self.grid_size = grid_size

    # 不做一维展平，分别按y、x两次gather，不用大索引
    def forward_twice(self, input, grid):
        B, C, H, W = input.shape
        Bg, Hout, Wout, _ = grid.shape
        assert B == Bg

        x = grid[..., 0]
        y = grid[..., 1]
        x = 0.5 * (x + 1) * (W - 1)
        y = 0.5 * (y + 1) * (H - 1)

        x_idx = torch.round(x).clamp(0, W-1).to(torch.int32)
        y_idx = torch.round(y).clamp(0, H-1).to(torch.int32)

        # input: [B,C,H,W]
        # 先取y维度
        tmp = torch.gather(input, dim=2, index=y_idx.unsqueeze(1).expand(-1,C,-1))
        # 再取x维度
        out = torch.gather(tmp, dim=3, index=x_idx.unsqueeze(1).expand(-1,C,-1))
        return out

    def forward_noGather(self, input, grid):
        B, C, H, W = input.shape
        Bg, Hout, Wout, _ = grid.shape
        assert B == Bg, f"batch size不匹配 input{B} grid{Bg}"

        x = grid[..., 0]
        y = grid[..., 1]

        x = 0.5 * (x + 1) * (W - 1)
        y = 0.5 * (y + 1) * (H - 1)

        x = torch.round(x).int()
        y = torch.round(y).int()

        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)

        batch_idx = torch.arange(B, device=input.device)[:, None, None]
        out = input[batch_idx, :, y, x].permute((0, 3, 1, 2))
        return out

    def forward_noGather_twice(self, input, grid):
        B, C, H, W = input.shape
        Bg, Hout, Wout, _ = grid.shape
        assert B == Bg, f"batch size不匹配 input{B} grid{Bg}"

        x = grid[..., 0]
        y = grid[..., 1]

        x = 0.5 * (x + 1) * (W - 1)
        y = 0.5 * (y + 1) * (H - 1)

        x = torch.round(x).int()
        y = torch.round(y).int()

        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)

        # ===================== 2. 分两次一维采样：先采x(列)，再采y(行) =====================
        # 【第一步：仅采样 x 方向（列维度 W），不使用 y】
        # input:      [1, C, H, W]
        # 沿最后一维 W 做一维索引 x
        feat_x = input[..., x]  # 输出 shape: [1, C, H, Hout, Wout]

        # 【第二步：仅采样 y 方向（行维度 H），不使用 x】
        # feat_x:     [1, C, H, Hout, Wout]
        # 沿第 2 维 H 做一维索引 y
        out = feat_x[:, :, y, :, :]  # 输出 shape: [1, C, Hout, Wout]

        return out

    def forward_origin(self, input, grid):
        B, C, H, W = input.shape
        Bg, Hout, Wout, _ = grid.shape
        assert B == Bg, f"batch size不匹配 input{B} grid{Bg}"

        x = grid[..., 0]
        y = grid[..., 1]

        x = 0.5 * (x + 1) * (W - 1)
        y = 0.5 * (y + 1) * (H - 1)

        x = torch.round(x).int()
        y = torch.round(y).int()

        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)

        idx = y * W + x   # 256 * 2500 too large
        idx = idx.reshape(B, 1, -1) #.expand(-1, C, -1)
        input_flat = input.reshape(B, C, H * W)
        out = torch.gather(input_flat, -1, idx).reshape(B, C, Hout, Wout)
        return out

    def fix_grid_integer_origin_label1(self, grid : torch.Tensor, C, H, W):
        B = grid.shape[0]

        x = grid[..., 0]
        y = grid[..., 1]

        x = 0.5 * (x + 1) * (W - 1)
        y = 0.5 * (y + 1) * (H - 1)

        x = torch.round(x).int()
        y = torch.round(y).int()

        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)

        idx = y * W + x   # 256 * 2500 too large
        idx = idx.reshape(B, 1, -1)#.expand(-1, C, -1)
        return idx

    def forward_origin_label1(self, input, idx):
        B, C, H, W = input.shape
        Hout, Wout = self.grid_size
        Bg = idx.shape[0]
        if not torch.onnx.is_in_onnx_export():
            assert B == Bg, f"batch size不匹配 input{B} grid{Bg}"

        if torch.onnx.is_in_onnx_export():
            input_flat = input.reshape(C, H * W)
            if C == 1:
                input_flat = input_flat[0]
            idx = idx.flatten()
            if len(input_flat.shape) == 2:
                out = input_flat[:, idx]
            else:
                out = input_flat[idx]
            out = out.reshape(C, Hout, Wout).unsqueeze(0)
        else:
            batch_idx = torch.arange(B, device=input.device, dtype=torch.int32)[:, None, None]
            input_flat = input.reshape(B, C, H * W)
            out = input_flat[batch_idx, :, idx].reshape(B, C, Hout, Wout)
        return out

class ViewTransformerNuscenes(nn.Module):
    """The view transform structure for converting image view to bev view.

    Args:
        num_views: Number for image views.
        bev_size: Bev size.
        grid_size: Grid size.
        mode: Mode for grid sample.
        padding_mode: Padding mode for grid sample.
        grid_quant_scale: Quanti scale for grid sample.
    """

    def __init__(
        self,
        num_views: int,
        bev_size: Tuple[float],
        grid_size: Tuple[float],
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        grid_quant_scale: float = 1 / 512,
        homo_key="ego2img",
        ocam_path: str = 'data/CarlaCollection/calib_results.txt',
        ocam_fov: float = 220,
        azimuth_range=[-math.radians(220/2), math.radians(220/2)],
        elevation_range=[-math.pi/4, math.pi/4],
        dbound = [0.5, 48.5, 0.5],
        useLidar2cam = False,
        useSpherical = False,
        feat_hw = [-1, -1],
        original_gridsample = False,
        feat_channels = -1,
        feat_scale = (1/16, 1/16),
        svd_rank = 16,
        num_tile = 4,
        learning_point = False,
        learning_point_version = '3',
        hidden_matmul = 256,
        learning_point_extrinsic = False,
        learning_point_upsampling_ratio = 1,
    ):
        super(ViewTransformerNuscenes, self).__init__()
        self.num_views = num_views
        self.original_gridsample = original_gridsample
        self.svd_rank = svd_rank
        self.bev_size = bev_size
        self.grid_size = grid_size
        self.num_tile = num_tile
        self.floatFs = FloatFunctional()

        self.tile_size = torch.ceil(torch.tensor(self.grid_size[0] * self.grid_size[1] / self.num_tile)).int().item()

        self.quant_stub = QuantStub(grid_quant_scale)
        if self.original_gridsample:
            self.grid_sample = hnn.GridSample(
                mode=mode,
                padding_mode=padding_mode,
            )
        else:
            self.grid_sample = GridSampleNearestReplace(grid_size = grid_size)
        self.feat_hw = feat_hw
        self.homo_key = homo_key
        self.ocam_path = ocam_path
        self.ocam_fov = ocam_fov
        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range
        self.dbound = dbound
        self.useLidar2cam = useLidar2cam
        self.useSpherical = useSpherical
        self.feat_scale = feat_scale
        self.learning_point = learning_point

        self.bev_quant = QuantStub()
        self.depth_bev_quant = QuantStub()
        self.learning_point_version = learning_point_version
        self.learning_point_upsampling_ratio = learning_point_upsampling_ratio
        self.learning_point_extrinsic = learning_point_extrinsic
        self.downsample_channel = 16

        self.cache_extrinsic = []
        self.feat_channels = feat_channels

        # if learning_point:
        #     self.merge_col = []
        #     for i in range(self.num_views):
        #         dataw2h = torch.rand((self.grid_size[0], self.grid_size[1])) * 2 - 1
        #         row_sumw2h = feat_hw[1]
        #         tmp = nn.Parameter(data = dataw2h / row_sumw2h, requires_grad = True)
        #         self.merge_col.append(tmp)


        # if learning_point:
        #     # if self.learning_point_upsampling_ratio > 1:
        #     #     feat_hw = [feat_hw[0] * learning_point_upsampling_ratio, feat_hw[1] * learning_point_upsampling_ratio]

        #     if self.learning_point_version == '1':
        #         # data1 = torch.rand((num_views, feat_hw[0], self.grid_size[0])) * 2 - 1
        #         # data2 = torch.rand((num_views, self.downsample_channel * feat_hw[1], self.downsample_channel * self.grid_size[1])) * 2 - 1

        #         # row_sum1 = feat_hw[0] * self.num_views
        #         # row_sum2 = self.downsample_channel * feat_hw[1] * self.num_views
        #         # self.param_point1 = nn.Parameter(data = data1 / row_sum1, requires_grad = True)
        #         # self.param_point2 = nn.Parameter(data = data2 / row_sum2, requires_grad = True)


        #         # ddata1 = torch.rand((num_views, feat_hw[0], self.grid_size[0])) * 2 - 1
        #         # ddata2 = torch.rand((num_views, self.downsample_channel * feat_hw[1], self.downsample_channel * self.grid_size[1])) * 2 - 1

        #         # drow_sum1 = 1  # feat_hw[0] * self.num_views
        #         # drow_sum2 = self.downsample_channel  # feat_hw[1] * self.num_views
        #         # self.dparam_point1 = nn.Parameter(data = ddata1 / drow_sum1, requires_grad = True)
        #         # self.dparam_point2 = nn.Parameter(data = ddata2 / drow_sum2, requires_grad = True)



        #         data1 = torch.randn((num_views, self.downsample_channel * feat_hw[0], self.downsample_channel * self.grid_size[0]))
        #         data2 = torch.randn((num_views, self.downsample_channel * feat_hw[1], self.downsample_channel * self.grid_size[1]))
        #         row_sum1 = torch.sum(data1, dim = (1), keepdim = True)
        #         row_sum1[row_sum1 == 0] = 1
        #         row_sum2 = torch.sum(data2, dim = (1), keepdim = True)
        #         row_sum2[row_sum2 == 0] = 1
        #         self.param_point1 = nn.Parameter(data = data1 / row_sum1, requires_grad = True)
        #         self.param_point2 = nn.Parameter(data = data2 / row_sum2, requires_grad = True)

        #         ddata1 = torch.randn((num_views, self.downsample_channel * feat_hw[0], self.downsample_channel * self.grid_size[0]))
        #         ddata2 = torch.randn((num_views, self.downsample_channel * feat_hw[1], self.downsample_channel * self.grid_size[1]))
        #         drow_sum1 = torch.sum(ddata1, dim = (1), keepdim = True)
        #         drow_sum1[drow_sum1 == 0] = 1
        #         drow_sum2 = torch.sum(ddata2, dim = (1), keepdim = True)
        #         drow_sum2[drow_sum2 == 0] = 1
        #         self.dparam_point1 = nn.Parameter(data = ddata1 / drow_sum1, requires_grad = True)
        #         self.dparam_point2 = nn.Parameter(data = ddata2 / drow_sum2, requires_grad = True)

        if learning_point:
            if self.learning_point_upsampling_ratio > 1:
                feat_hw = [feat_hw[0] * learning_point_upsampling_ratio, feat_hw[1] * learning_point_upsampling_ratio]
            self.num_layer = num_layer = 3
            if self.learning_point_version == '3muchmuchmuchbetter':
                data1 = torch.rand((num_views, feat_hw[0], hidden_matmul)) * 2 - 1
                data11 = torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1
                data111 = torch.rand((num_views, hidden_matmul, self.grid_size[0])) * 2 - 1
                data2 = torch.rand((num_views, feat_hw[1], hidden_matmul)) * 2 - 1
                data22 = torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1
                data222 = torch.rand((num_views, hidden_matmul, self.grid_size[1])) * 2 - 1

                data_1 = torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1
                data_11 = torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1
                data_111 = torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1
                data_111_final = torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1
                data_2 = torch.rand((num_views, feat_hw[0], feat_hw[0])) * 2 - 1
                data_22 = torch.rand((num_views, feat_hw[0], feat_hw[0])) * 2 - 1
                data_222 = torch.rand((num_views, feat_hw[0], feat_hw[0])) * 2 - 1

                data2_m = torch.rand((num_views, feat_hw[0], feat_hw[0])) * 2 - 1
                data22_m = torch.rand((num_views, feat_hw[0], feat_hw[0])) * 2 - 1
                data222_m = torch.rand((num_views, feat_hw[0], feat_hw[0])) * 2 - 1
                data1_m = torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1
                data11_m = torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1
                data111_m = torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1

                data1_col = [torch.rand((num_views, feat_hw[0], hidden_matmul)) * 2 - 1 for _ in range(num_layer)]
                data11_col = [torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1 for _ in range(num_layer)]
                data111_col = [torch.rand((num_views, hidden_matmul, self.grid_size[0])) * 2 - 1 for _ in range(num_layer)]
                data2_col = [torch.rand((num_views, feat_hw[1], hidden_matmul)) * 2 - 1 for _ in range(num_layer)]
                data22_col = [torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1 for _ in range(num_layer)]
                data222_col = [torch.rand((num_views, hidden_matmul, self.grid_size[1])) * 2 - 1 for _ in range(num_layer)]

                data_1_col = [torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1 for _ in range(num_layer)]
                data_11_col = [torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1 for _ in range(num_layer)]
                data_111_col = [torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1 for _ in range(num_layer)]
                data_2_col = [torch.rand((num_views, feat_hw[0], feat_hw[0])) * 2 - 1 for _ in range(num_layer)]
                data_22_col = [torch.rand((num_views, feat_hw[0], feat_hw[0])) * 2 - 1 for _ in range(num_layer)]
                data_222_col = [torch.rand((num_views, feat_hw[0], feat_hw[0])) * 2 - 1 for _ in range(num_layer)]

                # data1 = torch.randn((num_views, feat_hw[0], self.grid_size[0]))
                # data2 = torch.randn((num_views, feat_hw[1], self.grid_size[1]))
                # data1 = data1 / (data1.max() + data1.min().abs())
                # data2 = data2 / (data2.max() + data2.min().abs())

                # row_sum1 = torch.sum(data1, dim = (1), keepdim = True)
                # row_sum1[row_sum1 == 0] = 1
                # row_sum2 = torch.sum(data2, dim = (1), keepdim = True)
                # row_sum2[row_sum2 == 0] = 1

                row_sum1 = feat_hw[0] #* self.num_views
                row_sum11 = hidden_matmul #* self.num_views
                row_sum111 = hidden_matmul #* self.num_views
                row_sum2 = feat_hw[1] #* self.num_views
                row_sum22 = hidden_matmul #* self.num_views
                row_sum222 = hidden_matmul #* self.num_views
                row_sum_1 = self.grid_size[1] #* self.num_views
                row_sum_11 = self.grid_size[1] #* self.num_views
                row_sum_111 = self.grid_size[1] #* self.num_views
                row_sum_2 = feat_hw[0] #* self.num_views
                row_sum_22 = feat_hw[0] #* self.num_views
                row_sum_222 = feat_hw[0] #* self.num_views
                row_sum1_m = feat_hw[0] * self.num_views
                row_sum11_m = hidden_matmul * self.num_views
                row_sum111_m = hidden_matmul * self.num_views
                row_sum2_m = feat_hw[1] * self.num_views
                row_sum22_m = hidden_matmul * self.num_views
                row_sum222_m = hidden_matmul * self.num_views

                self.param_point1 = nn.Parameter(data = data1 / row_sum1, requires_grad = True)
                self.param_point11 = nn.Parameter(data = data11 / row_sum11, requires_grad = True)
                self.param_point111 = nn.Parameter(data = data111 / row_sum111, requires_grad = True)
                self.param_point2 = nn.Parameter(data = data2 / row_sum2, requires_grad = True)
                self.param_point22 = nn.Parameter(data = data22 / row_sum22, requires_grad = True)
                self.param_point222 = nn.Parameter(data = data222 / row_sum222, requires_grad = True)
                self.param_point_1 = nn.Parameter(data = data_1 / row_sum_1, requires_grad = True)
                self.param_point_11 = nn.Parameter(data = data_11 / row_sum_11, requires_grad = True)
                self.param_point_111 = nn.Parameter(data = data_111 / row_sum_111, requires_grad = True)
                self.param_point_111_final = nn.Parameter(data = data_111_final / row_sum_111, requires_grad = True)
                self.param_point_2 = nn.Parameter(data = data_2 / row_sum_2, requires_grad = True)
                self.param_point_22 = nn.Parameter(data = data_22 / row_sum_22, requires_grad = True)
                self.param_point_222 = nn.Parameter(data = data_222 / row_sum_222, requires_grad = True)

                self.param_point1_m = nn.Parameter(data = data1_m / row_sum1_m, requires_grad = True)
                self.param_point11_m = nn.Parameter(data = data11_m / row_sum11_m, requires_grad = True)
                self.param_point111_m = nn.Parameter(data = data111_m / row_sum111_m, requires_grad = True)
                self.param_point2_m = nn.Parameter(data = data2_m / row_sum2_m, requires_grad = True)
                self.param_point22_m = nn.Parameter(data = data22_m / row_sum22_m, requires_grad = True)
                self.param_point222_m = nn.Parameter(data = data222_m / row_sum222_m, requires_grad = True)

                self.param_point1_col = nn.Parameter(data = data1.clone() / row_sum1, requires_grad = True)
                self.param_point11_col = nn.Parameter(data = data11.clone() / row_sum11, requires_grad = True)
                self.param_point111_col = nn.Parameter(data = data111.clone() / row_sum111, requires_grad = True)
                self.param_point2_col = nn.Parameter(data = data2.clone() / row_sum2, requires_grad = True)
                self.param_point22_col = nn.Parameter(data = data22.clone() / row_sum22, requires_grad = True)
                self.param_point222_col = nn.Parameter(data = data222.clone() / row_sum222, requires_grad = True)
                self.param_point_1_col = nn.Parameter(data = data_1.clone() / row_sum_1, requires_grad = True)
                self.param_point_11_col =nn.Parameter(data = data_11.clone() / row_sum_11, requires_grad = True)
                self.param_point_111_col = nn.Parameter(data = data_111.clone() / row_sum_111, requires_grad = True)
                self.param_point_2_col = nn.Parameter(data = data_2.clone() / row_sum_2, requires_grad = True)
                self.param_point_22_col = nn.Parameter(data = data_22.clone() / row_sum_22, requires_grad = True)
                self.param_point_222_col = nn.Parameter(data = data_222.clone() / row_sum_222, requires_grad = True)


                self.param_point1_col1 = nn.Parameter(data = data1.clone() / row_sum1, requires_grad = True)
                self.param_point11_col1 = nn.Parameter(data = data11.clone() / row_sum11, requires_grad = True)
                self.param_point111_col1 = nn.Parameter(data = data111.clone() / row_sum111, requires_grad = True)
                self.param_point2_col1 = nn.Parameter(data = data2.clone() / row_sum2, requires_grad = True)
                self.param_point22_col1 = nn.Parameter(data = data22.clone() / row_sum22, requires_grad = True)
                self.param_point222_col1 = nn.Parameter(data = data222.clone() / row_sum222, requires_grad = True)
                self.param_point_1_col1 = nn.Parameter(data = data_1.clone() / row_sum_1, requires_grad = True)
                self.param_point_11_col1 =nn.Parameter(data = data_11.clone() / row_sum_11, requires_grad = True)
                self.param_point_111_col1 = nn.Parameter(data = data_111.clone() / row_sum_111, requires_grad = True)
                self.param_point_2_col1 = nn.Parameter(data = data_2.clone() / row_sum_2, requires_grad = True)
                self.param_point_22_col1 = nn.Parameter(data = data_22.clone() / row_sum_22, requires_grad = True)
                self.param_point_222_col1 = nn.Parameter(data = data_222.clone() / row_sum_222, requires_grad = True)

                self.param_point1_col2 = nn.Parameter(data = data1.clone() / row_sum1, requires_grad = True)
                self.param_point11_col2 = nn.Parameter(data = data11.clone() / row_sum11, requires_grad = True)
                self.param_point111_col2 = nn.Parameter(data = data111.clone() / row_sum111, requires_grad = True)
                self.param_point2_col2 = nn.Parameter(data = data2.clone() / row_sum2, requires_grad = True)
                self.param_point22_col2 = nn.Parameter(data = data22.clone() / row_sum22, requires_grad = True)
                self.param_point222_col2 = nn.Parameter(data = data222.clone() / row_sum222, requires_grad = True)
                self.param_point_1_col2 = nn.Parameter(data = data_1.clone() / row_sum_1, requires_grad = True)
                self.param_point_11_col2 =nn.Parameter(data = data_11.clone() / row_sum_11, requires_grad = True)
                self.param_point_111_col2 = nn.Parameter(data = data_111.clone() / row_sum_111, requires_grad = True)
                self.param_point_2_col2 = nn.Parameter(data = data_2.clone() / row_sum_2, requires_grad = True)
                self.param_point_22_col2 = nn.Parameter(data = data_22.clone() / row_sum_22, requires_grad = True)
                self.param_point_222_col2 = nn.Parameter(data = data_222.clone() / row_sum_222, requires_grad = True)



                ddata1 = torch.rand((num_views, feat_hw[0], hidden_matmul)) * 2 - 1
                ddata11 = torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1
                ddata111 = torch.rand((num_views, hidden_matmul, self.grid_size[0])) * 2 - 1
                ddata2 = torch.rand((num_views, feat_hw[1], hidden_matmul)) * 2 - 1
                ddata22 = torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1
                ddata222 = torch.rand((num_views, hidden_matmul, self.grid_size[1])) * 2 - 1

                # ddata1 = torch.randn((num_views, feat_hw[0], self.grid_size[0]))
                # ddata2 = torch.randn((num_views, feat_hw[1], self.grid_size[1]))
                # ddata1 = ddata1 / (ddata1.max() + ddata1.min().abs())
                # ddata2 = ddata2 / (ddata2.max() + ddata2.min().abs())

                # drow_sum1 = torch.sum(ddata1, dim = (1), keepdim = True)
                # drow_sum1[drow_sum1 == 0] = 1
                # drow_sum2 = torch.sum(ddata2, dim = (1), keepdim = True)
                # drow_sum2[drow_sum2 == 0] = 1

                drow_sum1 = 1 #feat_hw[0] * self.num_views
                drow_sum11 = 1 #hidden_matmul * self.num_views
                drow_sum111 = 1 #hidden_matmul * self.num_views
                drow_sum2 = 1 #feat_hw[1] * self.num_views
                drow_sum22 = 1 #hidden_matmul * self.num_views
                drow_sum222 = 1 #hidden_matmul * self.num_views
                self.dparam_point1 = nn.Parameter(data = ddata1 / drow_sum1, requires_grad = True)
                self.dparam_point11 = nn.Parameter(data = ddata11 / drow_sum11, requires_grad = True)
                self.dparam_point111 = nn.Parameter(data = ddata111 / drow_sum111, requires_grad = True)
                self.dparam_point2 = nn.Parameter(data = ddata2 / drow_sum2, requires_grad = True)
                self.dparam_point22 = nn.Parameter(data = ddata22 / drow_sum22, requires_grad = True)
                self.dparam_point222 = nn.Parameter(data = ddata222 / drow_sum222, requires_grad = True)


                data1_pc = torch.rand((num_views, 1, self.feat_channels, feat_hw[0], hidden_matmul)) * 2 - 1
                data11_pc = torch.rand((num_views, 1, self.feat_channels, hidden_matmul, hidden_matmul)) * 2 - 1
                data111_pc = torch.rand((num_views, 1, self.feat_channels, hidden_matmul, self.grid_size[0])) * 2 - 1
                data2_pc = torch.rand((num_views, 1, self.feat_channels, feat_hw[1], hidden_matmul)) * 2 - 1
                data22_pc = torch.rand((num_views, 1, self.feat_channels, hidden_matmul, hidden_matmul)) * 2 - 1
                data222_pc = torch.rand((num_views, 1, self.feat_channels, hidden_matmul, self.grid_size[1])) * 2 - 1

                data_1_pc = torch.rand((num_views, 1, self.feat_channels, self.grid_size[0], self.grid_size[1])) * 2 - 1
                data_11_pc = torch.rand((num_views, 1, self.feat_channels, self.grid_size[0], self.grid_size[1])) * 2 - 1
                data_111_pc = torch.rand((num_views, 1, self.feat_channels, self.grid_size[0], self.grid_size[1])) * 2 - 1
                data_2_pc = torch.rand((num_views, 1, self.feat_channels, feat_hw[0], feat_hw[0])) * 2 - 1
                data_22_pc = torch.rand((num_views, 1, self.feat_channels, feat_hw[0], feat_hw[0])) * 2 - 1
                data_222_pc = torch.rand((num_views, 1, self.feat_channels, feat_hw[0], feat_hw[0])) * 2 - 1

                self.param_point1_pc = nn.Parameter(data = data1_pc / row_sum1, requires_grad = True)
                self.param_point11_pc = nn.Parameter(data = data11_pc / row_sum11, requires_grad = True)
                self.param_point111_pc = nn.Parameter(data = data111_pc / row_sum111, requires_grad = True)
                self.param_point2_pc = nn.Parameter(data = data2_pc / row_sum2, requires_grad = True)
                self.param_point22_pc = nn.Parameter(data = data22_pc / row_sum22, requires_grad = True)
                self.param_point222_pc = nn.Parameter(data = data222_pc / row_sum222, requires_grad = True)
                self.param_point_1_pc = nn.Parameter(data = data_1_pc / row_sum_1, requires_grad = True)
                self.param_point_11_pc = nn.Parameter(data = data_11_pc / row_sum_11, requires_grad = True)
                self.param_point_111_pc = nn.Parameter(data = data_111_pc / row_sum_111, requires_grad = True)
                self.param_point_2_pc = nn.Parameter(data = data_2_pc / row_sum_2, requires_grad = True)
                self.param_point_22_pc = nn.Parameter(data = data_22_pc / row_sum_22, requires_grad = True)
                self.param_point_222_pc = nn.Parameter(data = data_222_pc / row_sum_222, requires_grad = True)

            elif self.learning_point_version == '2':
                data1 = torch.rand((num_views, feat_hw[0], hidden_matmul)) * 2 - 1
                data11 = torch.rand((num_views, hidden_matmul, self.grid_size[0])) * 2 - 1
                data2 = torch.rand((num_views, feat_hw[1], hidden_matmul)) * 2 - 1
                data22 = torch.rand((num_views, hidden_matmul, self.grid_size[1])) * 2 - 1

                row_sum1 = feat_hw[0] * self.num_views
                row_sum11 = hidden_matmul * self.num_views
                row_sum2 = feat_hw[1] * self.num_views
                row_sum22 = hidden_matmul * self.num_views
                self.param_point1 = nn.Parameter(data = data1 / row_sum1, requires_grad = True)
                self.param_point11 = nn.Parameter(data = data11 / row_sum11, requires_grad = True)
                self.param_point2 = nn.Parameter(data = data2 / row_sum2, requires_grad = True)
                self.param_point22 = nn.Parameter(data = data22 / row_sum22, requires_grad = True)


                ddata1 = torch.rand((num_views, feat_hw[0], hidden_matmul)) * 2 - 1
                ddata11 = torch.rand((num_views, hidden_matmul, self.grid_size[0])) * 2 - 1
                ddata2 = torch.rand((num_views, feat_hw[1], hidden_matmul)) * 2 - 1
                ddata22 = torch.rand((num_views, hidden_matmul, self.grid_size[1])) * 2 - 1

                drow_sum1 = 1 #feat_hw[0] * self.num_views
                drow_sum11 = 1 #hidden_matmul * self.num_views
                drow_sum2 = 1 #feat_hw[1] * self.num_views
                drow_sum22 = 1 #hidden_matmul * self.num_views
                self.dparam_point1 = nn.Parameter(data = ddata1 / drow_sum1, requires_grad = True)
                self.dparam_point11 = nn.Parameter(data = ddata11 / drow_sum11, requires_grad = True)
                self.dparam_point2 = nn.Parameter(data = ddata2 / drow_sum2, requires_grad = True)
                self.dparam_point22 = nn.Parameter(data = ddata22 / drow_sum22, requires_grad = True)
            elif self.learning_point_version == '1':
                data1 = torch.rand((num_views, feat_hw[0], self.grid_size[0])) * 2 - 1
                data2 = torch.rand((num_views, feat_hw[1], self.grid_size[1])) * 2 - 1

                row_sum1 = feat_hw[0] * self.num_views
                row_sum2 = feat_hw[1] * self.num_views
                self.param_point1 = nn.Parameter(data = data1 / row_sum1, requires_grad = True)
                self.param_point2 = nn.Parameter(data = data2 / row_sum2, requires_grad = True)


                ddata1 = torch.rand((num_views, feat_hw[0], self.grid_size[0])) * 2 - 1
                ddata2 = torch.rand((num_views, feat_hw[1], self.grid_size[1])) * 2 - 1

                drow_sum1 = 1 #feat_hw[0] * self.num_views
                drow_sum2 = 1 #feat_hw[1] * self.num_views
                self.dparam_point1 = nn.Parameter(data = ddata1 / drow_sum1, requires_grad = True)
                self.dparam_point2 = nn.Parameter(data = ddata2 / drow_sum2, requires_grad = True)

        self.intrinsic_net = nn.Linear(9, feat_hw[1])
        self.distortion_net = nn.Linear(4, feat_hw[1])
        self.cam2ego_net = nn.Linear(16, feat_hw[1])

        # if learning_point:
        #     if self.learning_point_upsampling_ratio > 1:
        #         feat_hw = [feat_hw[0] * learning_point_upsampling_ratio, feat_hw[1] * learning_point_upsampling_ratio]
        #     if self.learning_point_version == '3':
        #         # data1 = torch.rand((num_views, feat_hw[0], hidden_matmul)) * 2 - 1
        #         # data11 = torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1
        #         # data111 = torch.rand((num_views, hidden_matmul, self.grid_size[0])) * 2 - 1
        #         # data2 = torch.rand((num_views, feat_hw[1], hidden_matmul)) * 2 - 1
        #         # data22 = torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1
        #         # data222 = torch.rand((num_views, hidden_matmul, self.grid_size[1])) * 2 - 1

        #         data1 = torch.randn((num_views, feat_hw[0], hidden_matmul))
        #         data11 = torch.randn((num_views, hidden_matmul, hidden_matmul))
        #         data111 = torch.randn((num_views, hidden_matmul, self.grid_size[0]))
        #         data2 = torch.randn((num_views, feat_hw[1], hidden_matmul))
        #         data22 = torch.randn((num_views, hidden_matmul, hidden_matmul))
        #         data222 = torch.randn((num_views, hidden_matmul, self.grid_size[1]))

        #         row_sum1 = torch.sum(data1, dim = (1), keepdim = True)
        #         row_sum1[row_sum1 == 0] = 1
        #         row_sum11 = torch.sum(data11, dim = (1), keepdim = True)
        #         row_sum11[row_sum11 == 0] = 1
        #         row_sum111 = torch.sum(data111, dim = (1), keepdim = True)
        #         row_sum111[row_sum111 == 0] = 1
        #         row_sum2 = torch.sum(data2, dim = (1), keepdim = True)
        #         row_sum2[row_sum2 == 0] = 1
        #         row_sum22 = torch.sum(data22, dim = (1), keepdim = True)
        #         row_sum22[row_sum22 == 0] = 1
        #         row_sum222 = torch.sum(data222, dim = (1), keepdim = True)
        #         row_sum222[row_sum222 == 0] = 1

        #         # row_sum1 = feat_hw[0] * self.num_views
        #         # row_sum11 = hidden_matmul * self.num_views
        #         # row_sum111 = hidden_matmul * self.num_views
        #         # row_sum2 = feat_hw[1] * self.num_views
        #         # row_sum22 = hidden_matmul * self.num_views
        #         # row_sum222 = hidden_matmul * self.num_views
        #         # row_sum1 = row_sum11 = row_sum111 = row_sum2 = row_sum22 = row_sum222 = 1.0
        #         # tmp = data1 / row_sum1
        #         # kk = torch.sum(tmp[0, :, 0])
        #         # kk1 = torch.sum(tmp[0, :, 1])
        #         self.param_point1 = nn.Parameter(data = data1 / row_sum1, requires_grad = True)
        #         self.param_point11 = nn.Parameter(data = data11 / row_sum11, requires_grad = True)
        #         self.param_point111 = nn.Parameter(data = data111 / row_sum111, requires_grad = True)
        #         self.param_point2 = nn.Parameter(data = data2 / row_sum2, requires_grad = True)
        #         self.param_point22 = nn.Parameter(data = data22 / row_sum22, requires_grad = True)
        #         self.param_point222 = nn.Parameter(data = data222 / row_sum222, requires_grad = True)


        #         # ddata1 = torch.rand((num_views, feat_hw[0], hidden_matmul)) * 2 - 1
        #         # ddata11 = torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1
        #         # ddata111 = torch.rand((num_views, hidden_matmul, self.grid_size[0])) * 2 - 1
        #         # ddata2 = torch.rand((num_views, feat_hw[1], hidden_matmul)) * 2 - 1
        #         # ddata22 = torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1
        #         # ddata222 = torch.rand((num_views, hidden_matmul, self.grid_size[1])) * 2 - 1

        #         ddata1 = torch.randn((num_views, feat_hw[0], hidden_matmul))
        #         ddata11 = torch.randn((num_views, hidden_matmul, hidden_matmul))
        #         ddata111 = torch.randn((num_views, hidden_matmul, self.grid_size[0]))
        #         ddata2 = torch.randn((num_views, feat_hw[1], hidden_matmul))
        #         ddata22 = torch.randn((num_views, hidden_matmul, hidden_matmul))
        #         ddata222 = torch.randn((num_views, hidden_matmul, self.grid_size[1]))


        #         drow_sum1 = torch.sum(ddata1, dim = (1), keepdim = True)
        #         drow_sum1[drow_sum1 == 0] = 1
        #         drow_sum11 = torch.sum(ddata11, dim = (1), keepdim = True)
        #         drow_sum11[drow_sum11 == 0] = 1
        #         drow_sum111 = torch.sum(ddata111, dim = (1), keepdim = True)
        #         drow_sum111[drow_sum111 == 0] = 1
        #         drow_sum2 = torch.sum(ddata2, dim = (1), keepdim = True)
        #         drow_sum2[drow_sum2 == 0] = 1
        #         drow_sum22 = torch.sum(ddata22, dim = (1), keepdim = True)
        #         drow_sum22[drow_sum22 == 0] = 1
        #         drow_sum222 = torch.sum(ddata222, dim = (1), keepdim = True)
        #         drow_sum222[drow_sum222 == 0] = 1


        #         # drow_sum1 = feat_hw[0] * self.num_views
        #         # drow_sum11 = hidden_matmul * self.num_views
        #         # drow_sum111 = hidden_matmul * self.num_views
        #         # drow_sum2 = feat_hw[1] * self.num_views
        #         # drow_sum22 = hidden_matmul * self.num_views
        #         # drow_sum222 = hidden_matmul * self.num_views
        #         # drow_sum1 = drow_sum11 = drow_sum111 = drow_sum2 = drow_sum22 = drow_sum222 = 1.0
        #         self.dparam_point1 = nn.Parameter(data = ddata1 / drow_sum1, requires_grad = True)
        #         self.dparam_point11 = nn.Parameter(data = ddata11 / drow_sum11, requires_grad = True)
        #         self.dparam_point111 = nn.Parameter(data = ddata111 / drow_sum111, requires_grad = True)
        #         self.dparam_point2 = nn.Parameter(data = ddata2 / drow_sum2, requires_grad = True)
        #         self.dparam_point22 = nn.Parameter(data = ddata22 / drow_sum22, requires_grad = True)
        #         self.dparam_point222 = nn.Parameter(data = ddata222 / drow_sum222, requires_grad = True)
        #     elif self.learning_point_version == '2':
        #         # data1 = torch.rand((num_views, feat_hw[0], hidden_matmul)) * 2 - 1
        #         # data11 = torch.rand((num_views, hidden_matmul, self.grid_size[0])) * 2 - 1
        #         # data2 = torch.rand((num_views, feat_hw[1], hidden_matmul)) * 2 - 1
        #         # data22 = torch.rand((num_views, hidden_matmul, self.grid_size[1])) * 2 - 1

        #         data1 = torch.randn((num_views, feat_hw[0], hidden_matmul))
        #         data11 = torch.randn((num_views, hidden_matmul, self.grid_size[0]))
        #         data2 = torch.randn((num_views, feat_hw[1], hidden_matmul))
        #         data22 = torch.randn((num_views, hidden_matmul, self.grid_size[1]))

        #         # row_sum1 = feat_hw[0] * self.num_views
        #         # row_sum11 = hidden_matmul * self.num_views
        #         # row_sum2 = feat_hw[1] * self.num_views
        #         # row_sum22 = hidden_matmul * self.num_views
        #         row_sum1 = row_sum11 = row_sum2 = row_sum22 = 1.0
        #         self.param_point1 = nn.Parameter(data = data1 / row_sum1, requires_grad = True)
        #         self.param_point11 = nn.Parameter(data = data11 / row_sum11, requires_grad = True)
        #         self.param_point2 = nn.Parameter(data = data2 / row_sum2, requires_grad = True)
        #         self.param_point22 = nn.Parameter(data = data22 / row_sum22, requires_grad = True)


        #         # ddata1 = torch.rand((num_views, feat_hw[0], hidden_matmul)) * 2 - 1
        #         # ddata11 = torch.rand((num_views, hidden_matmul, self.grid_size[0])) * 2 - 1
        #         # ddata2 = torch.rand((num_views, feat_hw[1], hidden_matmul)) * 2 - 1
        #         # ddata22 = torch.rand((num_views, hidden_matmul, self.grid_size[1])) * 2 - 1

        #         ddata1 = torch.randn((num_views, feat_hw[0], hidden_matmul))
        #         ddata11 = torch.randn((num_views, hidden_matmul, self.grid_size[0]))
        #         ddata2 = torch.randn((num_views, feat_hw[1], hidden_matmul))
        #         ddata22 = torch.randn((num_views, hidden_matmul, self.grid_size[1]))

        #         # drow_sum1 = feat_hw[0] * self.num_views
        #         # drow_sum11 = hidden_matmul * self.num_views
        #         # drow_sum2 = feat_hw[1] * self.num_views
        #         # drow_sum22 = hidden_matmul * self.num_views
        #         drow_sum1 = drow_sum11 = drow_sum2 = drow_sum22 = 1.0
        #         self.dparam_point1 = nn.Parameter(data = ddata1 / drow_sum1, requires_grad = True)
        #         self.dparam_point11 = nn.Parameter(data = ddata11 / drow_sum11, requires_grad = True)
        #         self.dparam_point2 = nn.Parameter(data = ddata2 / drow_sum2, requires_grad = True)
        #         self.dparam_point22 = nn.Parameter(data = ddata22 / drow_sum22, requires_grad = True)
        #     elif self.learning_point_version == '1':
        #         # data1 = torch.rand((num_views, feat_hw[0], self.grid_size[0])) * 2 - 1
        #         # data2 = torch.rand((num_views, feat_hw[1], self.grid_size[1])) * 2 - 1

        #         data1 = torch.randn((num_views, feat_hw[0], self.grid_size[0]))
        #         data2 = torch.randn((num_views, feat_hw[1], self.grid_size[1]))

        #         # row_sum1 = feat_hw[0] * self.num_views
        #         # row_sum2 = feat_hw[1] * self.num_views
        #         row_sum1 = row_sum2 = 1
        #         self.param_point1 = nn.Parameter(data = data1 / row_sum1, requires_grad = True)
        #         self.param_point2 = nn.Parameter(data = data2 / row_sum2, requires_grad = True)

        #         # ddata1 = torch.rand((num_views, feat_hw[0], self.grid_size[0])) * 2 - 1
        #         # ddata2 = torch.rand((num_views, feat_hw[1], self.grid_size[1])) * 2 - 1

        #         ddata1 = torch.randn((num_views, feat_hw[0], self.grid_size[0]))
        #         ddata2 = torch.randn((num_views, feat_hw[1], self.grid_size[1]))

        #         # drow_sum1 = feat_hw[0] * self.num_views
        #         # drow_sum2 = feat_hw[1] * self.num_views
        #         drow_sum1 = drow_sum2 = 1
        #         self.dparam_point1 = nn.Parameter(data = ddata1 / drow_sum1, requires_grad = True)
        #         self.dparam_point2 = nn.Parameter(data = ddata2 / drow_sum2, requires_grad = True)

        self.ref_point = None

    def _gen_2d_points(self) -> Tensor:
        """Generate 2D points in the Bird's Eye View (BEV) space."""

        # Get the minimum and maximum x and y coordinates in the BEV space
        bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(
            self.bev_size
        )

        W = self.grid_size[0]
        H = self.grid_size[1]

        # Generate a tensor `x` containing the x-coordinates of the grid
        x = (
            torch.linspace(bev_min_x, bev_max_x, W)
            .reshape((1, W))
            .repeat(H, 1)
        ).double()
        # Generate a tensor `y` containing the y-coordinates of the grid
        y = (
            torch.linspace(bev_min_y, bev_max_y, H)
            .reshape((H, 1))
            .repeat(1, W)
        ).double()

        ones = torch.ones((H, W)).double()
        coords = torch.stack([x, y, ones], dim=-1)
        return coords

    def _convert_p2tensor(self, points: Any) -> Any:
        """Convert points to tensors.

        Args:
            points: The points to be converted.

        Returns:
            points: The converted points as tensors.
        """

        if isinstance(points, collections.abc.Sequence):
            for i in range(len(points)):
                points[i] = self._convert_p2tensor(points[i])
        if placeholder is not None and isinstance(points, placeholder):
            points = points.sample
        return points

    @fx_wrap()
    def forward(
        self, feats: Tensor, meta: Dict, compile_model: bool, **kwargs
    ) -> Any:
        """Perform the forward pass through the modules.

        Processes the input features and meta information to perform
        spatial transformation and generate reference points.

        Args:
            feats: The input features.
            meta: The meta information.
            compile_model: A flag indicating whether to compile the model.

        Returns:
            transformed_feats: The transformed features.
            points: The reference points.
        """

        feats = self._extract(feats)
        if self.learning_point:
            # if self.learning_point_extrinsic:
            #     with torch.no_grad():
            #         points = self.gen_reference_point(meta, feats)
            # return self.reduce_and_project_matrixVT(feats[0], feats[1], meta), None


            # sb = (feats[0].shape[0] // self.num_views)
            # cam2ego = meta['cam2ego'].view(sb * self.num_views, 16).float()
            # self.cam2ego = self.cam2ego_net(cam2ego)
            # distortion = meta["distortion"].float()
            # self.distortion = self.distortion_net(distortion[:, :4])
            # intrinsic = meta["camera_intrinsic"].view(sb * self.num_views, 9).float() / 1000
            # self.intrinsic = self.intrinsic_net(intrinsic)

            # self.proj_mats = self._precompute_geo_projection(meta["camera_intrinsic"], meta["distortion"].float(), meta['cam2ego'])
            # kk = torch.sum(proj_mats > 0)
            # lis = list(proj_mats.shape)
            # lis = lis[0] * lis[1] * lis[2]
            # return self._spatial_transfom(feats), None
            return self._spatial_transform_learn(feats), None
        else:
            if self.ref_point == None:
                if compile_model is True:
                    points = self._get_points_from_meta(meta)
                else:
                    with torch.no_grad():
                        points = self.gen_reference_point(meta, feats)
            else:
                points = (self.ref_point[0], self.ref_point[1])
            if self.original_gridsample:
                return self._spatial_transfom_origin(feats, points), None
            else:
                return self._spatial_transfom(feats, points), None
    
    def gen_reference_point(self, meta: Dict, feats: Tensor) -> Any:
        """Generate refrence points.

        Args:
            meta: A dictionary containing the input data.
            feats: The input for reference point generator.

        Returns:
            The Reference points.
        """
        with torch.no_grad():
            if len(self.cache_extrinsic) == 0 and self.learning_point_extrinsic:
                homography = self._get_homography(
                    meta, feats.shape[2:], self.homo_key
                )
                points = self._gen_reference_point_learn_point(homography, feats.shape[2:])
                self.cache_extrinsic = points
                return points

            homography = self._get_homography(
                meta, feats.shape[2:], self.homo_key
            )
            if self.original_gridsample:
                points = self._gen_reference_point_origin(homography, feats.shape[2:])
            else:
                points = self._gen_reference_point(homography, feats.shape[2:])
        return points

    def export_reference_points(
        self, meta: Dict, feat_hw: Tuple[int, int]
    ) -> Dict:
        """Export refrence points.

        Args:
            meta: A dictionary containing the input data.
            feat_hw: View transformer input shape
                     for generationg reference points.

        Returns:
            The Reference points.
        """
        homography = self._get_homography(meta, feat_hw, self.homo_key)
        if self.original_gridsample:
            points = self._gen_reference_point_origin(homography, feat_hw)
        else:
            points = self._gen_reference_point(homography, feat_hw)
        if not isinstance(points, Sequence):
            points = [points]

        ref_p_dict = {}
        for i, ref_p in enumerate(points):
            ref_p_dict[f"points{i}"] = ref_p
        return ref_p_dict

    def _extract(self, feats: Tensor) -> Tensor:
        """Extract the input features.

        Args:
            feats: The input features.

        Returns:
            feats: The input features.
        """
        return feats

    def _get_homography(
        self, meta: Dict, feat_hw: Tuple[int, int], homo_key="ego2img"
    ) -> Tensor:
        """Compute the homography matrix for mapping coordinates.

        Args:
            meta: The meta information.
            feat_hw: view transformer input shape
                     for generationg reference points.

        Returns:
            homography: The computed homography matrix.
        """

        # Get the ego2img homography matrix and the input
        # and original feature heights and widths
        homography = meta[homo_key]

        # orig_hw = meta["img"][0].shape[1:]
        # scales = (feat_hw[0] / orig_hw[0], feat_hw[1] / orig_hw[1])
        # view = np.eye(4)
        # view[0, 0] = scales[1]
        # view[1, 1] = scales[0]
        # view = torch.tensor(view).to(device=homography.device).double()

        # # Perform the matrix multiplication between
        # # the view transformation matrix and the homography matrix
        # homography = torch.matmul(view, homography.double())
        return homography

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        self.quant_stub.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "saturate": True},
            activation_calibration_qkwargs={"dtype": qint16, "saturate": True},
        )


@OBJECT_REGISTRY.register
class WrappingTransformerFisheye(ViewTransformerFisheye):
    """The IPM view transform for converting image view to bev view."""

    def __init__(self, **kwargs):
        super(WrappingTransformerFisheye, self).__init__(**kwargs)

    def _get_points_from_meta(self, meta: dict) -> Any:
        """Get the points from the meta information and convert them to tensors.

        Args:
            meta: The meta information.

        Returns:
            points: The points converted to tensors.
        """
        points = meta["points0"]
        return self._convert_p2tensor(points)

    def _gen_reference_point(
        self, homography: Tensor, feat_hw: Tuple[int, int] = None
    ) -> Tensor:
        """Generate and adjust the reference points.

        Args:
            homography: The homography matrix.
            feat_hw: View transformer input shape
                     for generationg reference points.

        Returns:
            new_coords: The generated and adjusted reference points.
        """

        # Generate 2D points coordinates in BEV space
        coords = self._gen_2d_points().to(device=homography.device)

        homography = homography[:, :3, (0, 1, 3)]

        # Perform mapping from BEV space to image space using homography matrix
        new_coords = []
        for homo in homography:
            new_coord = torch.matmul(coords, homo.permute((1, 0))).float()
            new_coords.append(new_coord)
        new_coords = torch.stack(new_coords)
        new_coords[..., 2] = torch.clamp(new_coords[..., 2], min=0.05)
        # Normalize the x and y coordinates by the z-coordinate
        X = new_coords[..., 0] / new_coords[..., 2]
        Y = new_coords[..., 1] / new_coords[..., 2]
        new_coords = torch.stack((X, Y), dim=-1)
        new_coords = adjust_coords(new_coords, self.grid_size)
        return new_coords

    def _spatial_transfom(self, feat: Tensor, points: Tensor) -> Tensor:
        """Apply spatial transformation to the input features.

        Args:
            feat: The input features.
            points: The reference points.

        Returns:
            fused_feats: The output features after spatial transformation.
        """
        if placeholder is not None and isinstance(points, placeholder):
            points = points.sample
        trans_feats = self.grid_sample(
            feat,
            self.quant_stub(points),
        )
        batch_size = int(trans_feats.shape[0] / self.num_views)

        if self.training or batch_size > 1:
            trans_feats = trans_feats.view(
                batch_size,
                self.num_views,
                trans_feats.shape[1],
                trans_feats.shape[2],
                trans_feats.shape[3],
            )
            fused_feats = self.floatFs.sum(trans_feats, keepdim=True, dim=1)
            fused_feats = fused_feats.view(
                batch_size,
                trans_feats.shape[2],
                trans_feats.shape[3],
                trans_feats.shape[4],
            )
        else:
            fused_feats = self.floatFs.sum(trans_feats, keepdim=True, dim=0)

        return fused_feats

    def fuse_model(self) -> None:
        pass

class ocamCameraTorch(OcamCamera):
    def __init__(self, filename, fov=360, show_flag=False):
        super(ocamCameraTorch, self).__init__(filename=filename, \
                                              fov = fov, \
                                              show_flag = show_flag)
        pass

    def world2camTorch(self, point3D):
        """ world2cam(point3D) projects a 3D point on to the image.
        If points are projected on the outside of the fov, return (-1,-1).
        Also, return (-1, -1), if point (x, y, z) = (0, 0, 0).
        The coordinate is different than that of the original OcamCalib.
        point3D coord: x:right direction, y:down direction, z:front direction
        point2D coord: x:row direction, y:col direction (OpenCV image coordinate).

        Parameters
        ----------
        point3D : numpy array or list([x, y, z])
            array of points in camera coordinate (3xN)

        Returns
        -------
        point2D : numpy array
            array of points in image (2xN)

        Examples
        --------
        >>> ocam = OcamCamera('./calib_results_0.txt')
        >>> ocam.world2cam([1,1,2.0]).tolist() # project a point on image
        [[1004.8294677734375], [1001.1594848632812]]
        >>> tmp = ocam.world2cam(np.random.rand(3, 10)) # project multiple points without error
        >>> ocam.world2cam([0,0,2.0]).tolist() # return optical center
        [[798.1757202148438], [794.3086547851562]]
        >>> ocam.world2cam([0,0,0]).tolist()
        [[-1.0], [-1.0]]
        """
        # in case of point3D = list([x,y,z])
        if isinstance(point3D, list):
            point3D = torch.tensor(point3D, dtype = torch.float32)
        if point3D.ndim == 1:
            point3D = point3D.unsqueeze(-1)
        assert point3D.shape[0] == 3

        # return value
        point2D = torch.zeros((2, point3D.shape[1]), dtype=torch.float32, device=point3D.device)

        norm = torch.sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1])
        valid_flag = (norm != 0)

        # optical center
        point2D[0][~valid_flag] = self._yc
        point2D[1][~valid_flag] = self._xc
        # point = (0, 0, 0)
        zero_flag = (point3D == 0).all(axis=0)
        point2D[0][zero_flag] = -1
        point2D[1][zero_flag] = -1

        # else
        theta = -torch.arctan(point3D[2][valid_flag] / norm[valid_flag])
        invnorm = 1 / norm[valid_flag]
        #     rho = np.array([element * theta ** i for (i, element) in enumerate(self._invpol)]).sum(axis=0) is slow
        for (i, element) in enumerate(self._invpol):
            if i == 0:
                rho = torch.full_like(theta, element)
                tmp_theta = theta.clone()
            else:
                rho += element * tmp_theta
                tmp_theta *= theta

        u = point3D[0][valid_flag] * invnorm * rho
        v = point3D[1][valid_flag] * invnorm * rho
        point2D_valid_0 = v * self._affine[2] + u + self._yc
        point2D_valid_1 = v * self._affine[0] + u * self._affine[1] + self._xc

        if self._fov < 360:
            # finally deal with points are outside of fov
            thresh_theta = torch.deg2rad(torch.tensor(self._fov / 2, dtype = torch.float32)) - np.pi / 2
            # set flag when  or point3D == (0, 0, 0)
            outside_flag = theta > thresh_theta
            point2D_valid_0[outside_flag] = -1
            point2D_valid_1[outside_flag] = -1

        point2D[0][valid_flag] = point2D_valid_0
        point2D[1][valid_flag] = point2D_valid_1

        return point2D

@OBJECT_REGISTRY.register
class LSSTransformerNuscenes(ViewTransformerFisheye):
    """The Lift-Splat-Shoot view transform for converting image view to bev view.

    Args:
        in_channels: In channel of feature.
        feat_channels: Feature channel of lift.
        z_range: The range of Z for bev coordarin.
        num_points: Num points for each voxel.
        depth: Depth value.
        mode: Mode for grid sample.
        padding_mode: Padding mode for grid sample.
        dgrid_quant_scale: Quanti scale for depth grid sample.

    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        z_range: Tuple[float] = (-10.0, 10.0),
        num_points: int = 10,
        depth: int = 60,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        depth_grid_quant_scale: float = 1 / 512,
        use_vtv2: bool = False,
        cal_minmax: bool = True,
        **kwargs,
    ):
        super(LSSTransformerFisheye, self).__init__(
            mode=mode, padding_mode=padding_mode, 
            feat_channels = feat_channels, **kwargs
        )
        self.depth = depth
        self.feat_channels = feat_channels
        self.z_range = z_range
        self.depth_net = ConvModule2d(
            in_channels=in_channels,
            out_channels=depth,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.feat_net = ConvModule2d(
            in_channels=in_channels,
            # out_channels=feat_channels,
            out_channels=feat_channels * self.num_layer,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.softmax = nn.Softmax(dim=1)
        self.dquant_stub = QuantStub(depth_grid_quant_scale)
        if self.original_gridsample:
            self.dgrid_sample = hnn.GridSample(
                mode=mode,
                padding_mode=padding_mode,
            )
        self.use_vtv2 = use_vtv2
        self.cal_minmax = cal_minmax
        self.num_points = num_points
        self.relu = nn.ReLU()

        self.linear3d = nn.Linear(self.feat_hw[0] * self.feat_hw[1], self.grid_size[0] * self.grid_size[1])

        self.feat_downsample = ConvModule2d(self.feat_channels, self.downsample_channel, (1, 1),  
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.downsample_channel),
                                                act_layer=nn.ReLU(inplace=True),)
        self.depth_downsample = ConvModule2d(self.depth, self.downsample_channel, (1, 1),  
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.downsample_channel),
                                                act_layer=nn.ReLU(inplace=True),)
        # self.ds_f0 = nn.ModuleList([ConvModule2d(self.downsample_channel, self.feat_channels, (1, 1), 
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        # self.ds_d0 = nn.ModuleList([ConvModule2d(self.downsample_channel, self.depth, (1, 1), 
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])

        self.fbn0 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn00 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn000 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn1 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn11 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn111 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])

        self.fbn0_col = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn00_col = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn000_col = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn1_col = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn11_col = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn111_col = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])


        self.fbn0_col1 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn00_col1 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn000_col1 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn1_col1 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn11_col1 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn111_col1 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])

        self.fbn0_col2 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn00_col2 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn000_col2 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn1_col2 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn11_col2 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn111_col2 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        # self.fbn0_col = nn.ModuleList([ nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)]) for _ in range(self.num_layer) ] )
        # self.fbn00_col = nn.ModuleList([ nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)]) for _ in range(self.num_layer) ] )
        # self.fbn000_col = nn.ModuleList([ nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)]) for _ in range(self.num_layer) ] )
        self.fbn_0 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn_00 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.fbn_000 = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        # self.fbn1_col = nn.ModuleList([ nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)]) for _ in range(self.num_layer) ])
        # self.fbn11_col = nn.ModuleList([ nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)]) for _ in range(self.num_layer) ])
        # self.fbn111_col = nn.ModuleList([ nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1), 
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)]) for _ in range(self.num_layer) ])
        # self.dbn0 = nn.ModuleList([ConvModule2d(self.depth, self.depth, (1, 1),  
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        # self.dbn00 = nn.ModuleList([ConvModule2d(self.depth, self.depth, (1, 1),  
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        # self.dbn000 = nn.ModuleList([ConvModule2d(self.depth, self.depth, (1, 1),  
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        # self.dbn1 = nn.ModuleList([ConvModule2d(self.depth, self.depth, (1, 1),  
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        # self.dbn11 = nn.ModuleList([ConvModule2d(self.depth, self.depth, (1, 1),  
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        # self.dbn111 = nn.ModuleList([ConvModule2d(self.depth, self.depth, (1, 1),  
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        
        # self.bnMerge = nn.ModuleList([ConvModule2d(self.feat_channels, self.feat_channels, (1, 1),  
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        # self.bnMergeFD = nn.ModuleList([ConvModule2d(self.feat_channels * 2, self.feat_channels, (1, 1),  
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
        #                                         act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])

        # self.fbn0All = ConvModule2d(self.feat_channels * self.num_views, self.feat_channels * self.num_views, (1, 1), 
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels * self.num_views),
        #                                         act_layer=nn.ReLU(inplace=True),)
        # self.fbn1All = ConvModule2d(self.feat_channels * self.num_views, self.feat_channels * self.num_views, (1, 1), 
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.feat_channels * self.num_views),
        #                                         act_layer=nn.ReLU(inplace=True),)
        # self.dbn0All = ConvModule2d(self.depth * self.num_views, self.depth * self.num_views, (1, 1),  
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth * self.num_views),
        #                                         act_layer=nn.ReLU(inplace=True),)
        # self.dbn1All = ConvModule2d(self.depth * self.num_views, self.depth * self.num_views, (1, 1),  
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth * self.num_views),
        #                                         act_layer=nn.ReLU(inplace=True),)
        self.bnMergeFD_col = nn.ModuleList([ConvModule2d(self.feat_channels * self.num_layer, self.feat_channels, (1, 1),  
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.bnMergeALL = nn.Sequential(
                                ConvModule2d(self.feat_channels * self.num_views, self.depth, (3, 3),  
                                                stride=1, padding=1, norm_layer=nn.BatchNorm2d(self.depth),
                                                act_layer=nn.ReLU(inplace=True),),
                                # ConvModule2d(self.feat_channels * self.num_views, self.feat_channels * self.num_views, (3, 3),  
                                #                 stride=1, padding=1, norm_layer=nn.BatchNorm2d(self.feat_channels * self.num_views),
                                #                 act_layer=nn.ReLU(inplace=True),),
                                # ConvModule2d(self.feat_channels * self.num_views, self.depth, (1, 1),  
                                #                 stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
                                #                 act_layer=nn.ReLU(inplace=True),),
                                # ConvModule2d(self.depth, self.depth, (1, 1),  
                                #                 stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
                                #                 act_layer=nn.ReLU(inplace=True),)
                                )
        self.bnMergeALL_col =   ConvModule2d(self.feat_channels * self.num_views, self.depth, (3, 3),  
                                             stride=1, padding=1, norm_layer=nn.BatchNorm2d(self.depth),
                                             act_layer=nn.ReLU(inplace=True))
        self.bnMergeALL_col_views =   ConvModule2d(self.feat_channels * self.num_views * self.num_layer, self.depth, (3, 3),  
                                             stride=1, padding=1, norm_layer=nn.BatchNorm2d(self.depth),
                                             act_layer=nn.ReLU(inplace=True))
        # self.bnMergeFD_ALL = ConvModule2d(self.feat_channels * self.num_views * 2, self.depth, (3, 3),  
        #                                         stride=1, padding=1, norm_layer=nn.BatchNorm2d(self.depth),
        #                                         act_layer=nn.ReLU(inplace=True),)
        # self.resnetAdd = ConvModule2d(self.feat_channels * self.num_views, self.depth, (1, 1),  
        #                                         stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
        #                                         act_layer=nn.ReLU(inplace=True),)
        if os.path.exists(self.ocam_path):
            self.omni_ocam = ocamCameraTorch(filename=self.ocam_path, fov=self.ocam_fov)

    def gen_reference_point(self, meta: Dict, feats: List[Tensor]) -> Any:
        """Generate refrence points.

        Args:
            meta: A dictionary containing the input data.
            feats: The input for reference point generator.

        Returns:
            The Reference points.
        """
        return super().gen_reference_point(meta, feats[0])

    def _get_points_from_meta(self, meta: Dict) -> List[Tensor]:
        """Extract points from metadata dictionary.

        Args:
            meta: Metadata dictionary.

        Returns:
            points: List of extracted points as converted tensors.
        """
        points = []
        for k in meta.keys():
            if k.startswith("points"):
                points.append(self._convert_p2tensor(meta[k]))
        assert len(points) == 2
        return points

    def _extract(
        self, feats: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Extract features and depth using the feature tensor.

        Args:
            feats: Feature tensor.

        Returns:
            Tuple containing the extracted features and depth.
        """

        new_feats = []

        depth = None #self.softmax(self.depth_net(feats))
        # depth = self.depth_net(feats)

        new_feats = self.feat_net(feats)
        return new_feats, depth

    def _spatial_transform_learn(self, feats: Tensor) -> Tensor:
        """Apply spatial transformation to the features using the given points.

        Args:
            feats: Tuple of feature tensor and depth tensor.
            points: Tuple of feature points and depth points.

        Returns:
            The transformed feature tensor.
        """
        feat, dfeat = feats # torch.Size([8, 64, 50, 50])      torch.Size([8, 64, 50, 50])
        if self.learning_point_upsampling_ratio > 1:
            feat = torch.nn.functional.interpolate(feat, scale_factor=self.learning_point_upsampling_ratio, \
                                                   mode = "bilinear", recompute_scale_factor=True,)
            dfeat = torch.nn.functional.interpolate(dfeat, scale_factor=self.learning_point_upsampling_ratio, \
                                                    mode = "bilinear", recompute_scale_factor=True,)
        B = feat.shape[0] // self.num_views
        C, H, W = feat.shape[1:]

        if self.training or B > 1:
            feat = feat.view(B, self.num_views, C, H, W)
            dfeat = dfeat.view(B, self.num_views, -1, H, W)
            feat = feat.permute(1, 0, 2, 3, 4).contiguous()
            dfeat = dfeat.permute(1, 0, 2, 3, 4).contiguous()
        else:
            feat = feat.contiguous()
            dfeat = dfeat.contiguous()

        if feat.device != self.param_point2_col[0].device:
            for i in range(self.num_layer):
                self.param_point2_col[i] = self.param_point2_col[i].to(device = feat.device)
                self.param_point22_col[i] = self.param_point22_col[i].to(device = feat.device)
                self.param_point222_col[i] = self.param_point222_col[i].to(device = feat.device)
                self.param_point1_col[i] = self.param_point1_col[i].to(device = feat.device)
                self.param_point11_col[i] = self.param_point11_col[i].to(device = feat.device)
                self.param_point111_col[i] = self.param_point111_col[i].to(device = feat.device)
                self.param_point_2_col[i] = self.param_point_2_col[i].to(device = feat.device)
                self.param_point_22_col[i] = self.param_point_22_col[i].to(device = feat.device)
                self.param_point_222_col[i] = self.param_point_222_col[i].to(device = feat.device)
                self.param_point_1_col[i] = self.param_point_1_col[i].to(device = feat.device)
                self.param_point_11_col[i] = self.param_point_11_col[i].to(device = feat.device)
                self.param_point_111_col[i] = self.param_point_111_col[i].to(device = feat.device)
        # if self.merge_col[0].device != feat.device:
        #     for i in range(self.num_views):
        #         self.merge_col[i] = self.merge_col[i].to(device = feat.device)

        flat_bev_col = []
        if self.learning_point_version == '3muchbetter':                # 333SplitMatrix_SameWeight 3muchbetter
            for i in range(self.num_views):
                img_feat = feat[i]
                depth_feat = dfeat[i]
                if len(img_feat.shape) == 3:
                    img_feat = img_feat.unsqueeze(0)
                    depth_feat = depth_feat.unsqueeze(0)

                flat_bev00 = torch.matmul( img_feat, self.param_point2[i] )
                flat_bev = self.fbn0[i](flat_bev00)
                flat_bev = torch.matmul(self.param_point_2[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point22[i] )
                flat_bev = self.fbn00[i](flat_bev)
                flat_bev = torch.matmul(self.param_point_22[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point222[i] )
                flat_bev = self.fbn000[i](flat_bev) + flat_bev00
                flat_bev = flat_bev.permute((0, 1, 3, 2))
                flat_bev = torch.matmul(self.param_point_222[i], flat_bev)

                flat_bev11 = torch.matmul( flat_bev, self.param_point1[i] )
                flat_bev = self.fbn1[i](flat_bev11)
                flat_bev = torch.matmul(self.param_point_1[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point11[i] )
                flat_bev = self.fbn11[i](flat_bev)
                flat_bev = torch.matmul(self.param_point_11[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point111[i] )
                flat_bev = self.fbn111[i](flat_bev) + flat_bev11
                flat_bev = flat_bev.permute((0, 1, 3, 2))
                flat_bev = torch.matmul(self.param_point_111[i], flat_bev)
                flat_bev = torch.matmul(flat_bev, self.param_point_111_final[i])

                # flat_depth_bev00 = torch.matmul( depth_feat, self.param_point2[i] )
                # flat_depth_bev = self.dbn0[i](flat_depth_bev00)
                # flat_depth_bev = torch.matmul(self.param_point_2[i], flat_depth_bev)

                # flat_depth_bev = torch.matmul( flat_depth_bev, self.param_point22[i] )
                # flat_depth_bev = self.dbn00[i](flat_depth_bev)
                # flat_depth_bev = torch.matmul(self.param_point_22[i], flat_depth_bev)

                # flat_depth_bev = torch.matmul( flat_depth_bev, self.param_point222[i] )
                # flat_depth_bev = self.dbn000[i](flat_depth_bev) + flat_depth_bev00
                # flat_depth_bev = flat_depth_bev.permute((0, 1, 3, 2))
                # flat_depth_bev = torch.matmul(self.param_point_222[i], flat_depth_bev)

                # flat_depth_bev11 = torch.matmul( flat_depth_bev, self.param_point1[i] )
                # flat_depth_bev = self.dbn1[i](flat_depth_bev11)
                # flat_depth_bev = torch.matmul(self.param_point_1[i], flat_depth_bev)

                # flat_depth_bev = torch.matmul( flat_depth_bev, self.param_point11[i] )
                # flat_depth_bev = self.dbn11[i](flat_depth_bev)
                # flat_depth_bev = torch.matmul(self.param_point_11[i], flat_depth_bev)

                # flat_depth_bev = torch.matmul( flat_depth_bev, self.param_point111[i] )
                # flat_depth_bev = self.dbn111[i](flat_depth_bev) + flat_depth_bev11
                # flat_depth_bev = flat_depth_bev.permute((0, 1, 3, 2))
                # flat_depth_bev = torch.matmul(self.param_point_111[i], flat_depth_bev)

                # flat_bev_col.append(flat_bev + flat_depth_bev)
                bev_img_fp = self.bev_quant(flat_bev)
                # bev_depth_fp = self.depth_bev_quant(flat_depth_bev)

                # bev_img_fp = torch.matmul(self.merge_col[i], bev_img_fp)
                # bev_depth_fp = torch.matmul(self.merge_col[i], bev_depth_fp)
                
                # fused = bev_img_fp * bev_depth_fp
                # if len(fused.shape)==3:
                #     fused = fused.unsqueeze(0)
                fused = bev_img_fp #self.bnMergeFD[i](torch.concat([bev_img_fp, bev_depth_fp], dim = 1))
                flat_bev_col.append(fused)# / (self.num_views))
        elif self.learning_point_version == '3muchmuchbetter':                # 333SplitMatrix_SameWeight 3muchmuchbetter
            for i in range(self.num_views):
                img_feat = feat[i]
                depth_feat = dfeat[i]
                if len(img_feat.shape) == 3:
                    img_feat = img_feat.unsqueeze(0)
                    depth_feat = depth_feat.unsqueeze(0)

                flat_bev00 = torch.matmul( img_feat, self.param_point2[i] )
                # flat_bev00 = torch.matmul(self.param_point2_m[i], flat_bev00)
                flat_bev = self.fbn0[i](flat_bev00)
                flat_bev = torch.matmul(self.param_point_2[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point22[i] )
                # flat_bev = torch.matmul(self.param_point22_m[i], flat_bev)
                flat_bev = self.fbn00[i](flat_bev)
                flat_bev = torch.matmul(self.param_point_22[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point222[i] )
                # flat_bev = torch.matmul( self.param_point222_m[i], flat_bev)
                flat_bev = self.fbn000[i](flat_bev) + flat_bev00
                flat_bev = torch.matmul(self.param_point_222[i], flat_bev)
                flat_bev = flat_bev.permute((0, 1, 3, 2))

                flat_bev11 = torch.matmul( flat_bev, self.param_point1[i] )
                # flat_bev11 = torch.matmul( self.param_point1_m[i], flat_bev11)
                flat_bev = self.fbn1[i](flat_bev11)
                flat_bev = torch.matmul(self.param_point_1[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point11[i] )
                # flat_bev = torch.matmul( self.param_point11_m[i], flat_bev)
                flat_bev = self.fbn11[i](flat_bev)
                flat_bev = torch.matmul(self.param_point_11[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point111[i] )
                # flat_bev = torch.matmul( self.param_point111_m[i], flat_bev)
                flat_bev = self.fbn111[i](flat_bev) + flat_bev11
                flat_bev = torch.matmul(self.param_point_111[i], flat_bev)
                flat_bev = flat_bev.permute((0, 1, 3, 2))
                # flat_bev = torch.matmul(flat_bev, self.param_point_111_final[i])

                # flat_depth_bev00 = torch.matmul( depth_feat, self.param_point2[i] )
                # flat_depth_bev = self.dbn0[i](flat_depth_bev00)
                # flat_depth_bev = torch.matmul(self.param_point_2[i], flat_depth_bev)

                # flat_depth_bev = torch.matmul( flat_depth_bev, self.param_point22[i] )
                # flat_depth_bev = self.dbn00[i](flat_depth_bev)
                # flat_depth_bev = torch.matmul(self.param_point_22[i], flat_depth_bev)

                # flat_depth_bev = torch.matmul( flat_depth_bev, self.param_point222[i] )
                # flat_depth_bev = self.dbn000[i](flat_depth_bev) + flat_depth_bev00
                # flat_depth_bev = flat_depth_bev.permute((0, 1, 3, 2))
                # flat_depth_bev = torch.matmul(self.param_point_222[i], flat_depth_bev)

                # flat_depth_bev11 = torch.matmul( flat_depth_bev, self.param_point1[i] )
                # flat_depth_bev = self.dbn1[i](flat_depth_bev11)
                # flat_depth_bev = torch.matmul(self.param_point_1[i], flat_depth_bev)

                # flat_depth_bev = torch.matmul( flat_depth_bev, self.param_point11[i] )
                # flat_depth_bev = self.dbn11[i](flat_depth_bev)
                # flat_depth_bev = torch.matmul(self.param_point_11[i], flat_depth_bev)

                # flat_depth_bev = torch.matmul( flat_depth_bev, self.param_point111[i] )
                # flat_depth_bev = self.dbn111[i](flat_depth_bev) + flat_depth_bev11
                # flat_depth_bev = flat_depth_bev.permute((0, 1, 3, 2))
                # flat_depth_bev = torch.matmul(self.param_point_111[i], flat_depth_bev)

                # flat_bev_col.append(flat_bev + flat_depth_bev)
                bev_img_fp = self.bev_quant(flat_bev)
                # bev_depth_fp = self.depth_bev_quant(flat_depth_bev)

                # bev_img_fp = torch.matmul(self.merge_col[i], bev_img_fp)
                # bev_depth_fp = torch.matmul(self.merge_col[i], bev_depth_fp)
                
                # fused = bev_img_fp * bev_depth_fp
                # if len(fused.shape)==3:
                #     fused = fused.unsqueeze(0)
                fused = bev_img_fp #self.bnMergeFD[i](torch.concat([bev_img_fp, bev_depth_fp], dim = 1))
                flat_bev_col.append(fused)# / (self.num_views))
        elif self.learning_point_version == '3muchmuchmuchbetter':                # 333SplitMatrix_SameWeight
            for i in range(self.num_views):
                img_feat = feat[i]
                depth_feat = dfeat[i]
                if len(img_feat.shape) == 3:
                    img_feat = img_feat.unsqueeze(0)
                    depth_feat = depth_feat.unsqueeze(0)
                collect = []
                # for j in range(self.num_layer):
                flat_bev00 = torch.matmul( img_feat[:, :64], self.param_point2_col[i] )
                # flat_bev00 = torch.matmul(self.param_point2_m[i], flat_bev00)
                flat_bev = self.fbn0_col1[i](flat_bev00)
                flat_bev = torch.matmul(self.param_point_2_col[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point22_col[i] )
                # flat_bev = torch.matmul(self.param_point22_m[i], flat_bev)
                flat_bev = self.fbn00_col1[i](flat_bev)
                flat_bev = torch.matmul(self.param_point_22_col[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point222_col[i] )
                # flat_bev = torch.matmul( self.param_point222_m[i], flat_bev)
                flat_bev = self.fbn000_col1[i](flat_bev) + flat_bev00
                flat_bev = torch.matmul(self.param_point_222_col[i], flat_bev)
                flat_bev = flat_bev.permute((0, 1, 3, 2))

                flat_bev11 = torch.matmul( flat_bev, self.param_point1_col[i] )
                # flat_bev11 = torch.matmul( self.param_point1_m[i], flat_bev11)
                flat_bev = self.fbn1_col1[i](flat_bev11)
                flat_bev = torch.matmul(self.param_point_1_col[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point11_col[i] )
                # flat_bev = torch.matmul( self.param_point11_m[i], flat_bev)
                flat_bev = self.fbn11_col1[i](flat_bev)
                flat_bev = torch.matmul(self.param_point_11_col[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point111_col[i] )
                # flat_bev = torch.matmul( self.param_point111_m[i], flat_bev)
                flat_bev = self.fbn111_col1[i](flat_bev) + flat_bev11
                flat_bev = torch.matmul(self.param_point_111_col[i], flat_bev)
                flat_bev_ret0 = flat_bev.permute((0, 1, 3, 2))



                flat_bev00 = torch.matmul( img_feat[:, 64:64*2], self.param_point2_col1[i] )
                # flat_bev00 = torch.matmul(self.param_point2_m[i], flat_bev00)
                flat_bev = self.fbn0_col[i](flat_bev00)
                flat_bev = torch.matmul(self.param_point_2_col1[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point22_col1[i] )
                # flat_bev = torch.matmul(self.param_point22_m[i], flat_bev)
                flat_bev = self.fbn00_col[i](flat_bev)
                flat_bev = torch.matmul(self.param_point_22_col1[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point222_col1[i] )
                # flat_bev = torch.matmul( self.param_point222_m[i], flat_bev)
                flat_bev = self.fbn000_col[i](flat_bev) + flat_bev00
                flat_bev = torch.matmul(self.param_point_222_col1[i], flat_bev)
                flat_bev = flat_bev.permute((0, 1, 3, 2))

                flat_bev11 = torch.matmul( flat_bev, self.param_point1_col1[i] )
                # flat_bev11 = torch.matmul( self.param_point1_m[i], flat_bev11)
                flat_bev = self.fbn1_col[i](flat_bev11)
                flat_bev = torch.matmul(self.param_point_1_col1[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point11_col1[i] )
                # flat_bev = torch.matmul( self.param_point11_m[i], flat_bev)
                flat_bev = self.fbn11_col[i](flat_bev)
                flat_bev = torch.matmul(self.param_point_11_col1[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point111_col1[i] )
                # flat_bev = torch.matmul( self.param_point111_m[i], flat_bev)
                flat_bev = self.fbn111_col[i](flat_bev) + flat_bev11
                flat_bev = torch.matmul(self.param_point_111_col1[i], flat_bev)
                flat_bev_ret1 = flat_bev.permute((0, 1, 3, 2))



                flat_bev00 = torch.matmul( img_feat[:, 64*2:], self.param_point2_col2[i] )
                # flat_bev00 = torch.matmul(self.param_point2_m[i], flat_bev00)
                flat_bev = self.fbn0_col2[i](flat_bev00)
                flat_bev = torch.matmul(self.param_point_2_col2[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point22_col2[i] )
                # flat_bev = torch.matmul(self.param_point22_m[i], flat_bev)
                flat_bev = self.fbn00_col2[i](flat_bev)
                flat_bev = torch.matmul(self.param_point_22_col2[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point222_col2[i] )
                # flat_bev = torch.matmul( self.param_point222_m[i], flat_bev)
                flat_bev = self.fbn000_col2[i](flat_bev) + flat_bev00
                flat_bev = torch.matmul(self.param_point_222_col2[i], flat_bev)
                flat_bev = flat_bev.permute((0, 1, 3, 2))

                flat_bev11 = torch.matmul( flat_bev, self.param_point1_col2[i] )
                # flat_bev11 = torch.matmul( self.param_point1_m[i], flat_bev11)
                flat_bev = self.fbn1_col2[i](flat_bev11)
                flat_bev = torch.matmul(self.param_point_1_col2[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point11_col2[i] )
                # flat_bev = torch.matmul( self.param_point11_m[i], flat_bev)
                flat_bev = self.fbn11_col2[i](flat_bev)
                flat_bev = torch.matmul(self.param_point_11_col2[i], flat_bev)

                flat_bev = torch.matmul( flat_bev, self.param_point111_col2[i] )
                # flat_bev = torch.matmul( self.param_point111_m[i], flat_bev)
                flat_bev = self.fbn111_col2[i](flat_bev) + flat_bev11
                flat_bev = torch.matmul(self.param_point_111_col2[i], flat_bev)
                flat_bev_ret2 = flat_bev.permute((0, 1, 3, 2))

                    # flat_bev00 = torch.matmul( img_feat, self.param_point2[i] )
                    # # flat_bev00 = torch.matmul(self.param_point2_m[i], flat_bev00)
                    # flat_bev = self.fbn0[i](flat_bev00)
                    # flat_bev = torch.matmul(self.param_point_2[i], flat_bev)

                    # flat_bev = torch.matmul( flat_bev, self.param_point22[i] )
                    # # flat_bev = torch.matmul(self.param_point22_m[i], flat_bev)
                    # flat_bev = self.fbn00[i](flat_bev)
                    # flat_bev = torch.matmul(self.param_point_22[i], flat_bev)

                    # flat_bev = torch.matmul( flat_bev, self.param_point222[i] )
                    # # flat_bev = torch.matmul( self.param_point222_m[i], flat_bev)
                    # flat_bev = self.fbn000[i](flat_bev) + flat_bev00
                    # flat_bev = torch.matmul(self.param_point_222[i], flat_bev)
                    # flat_bev = flat_bev.permute((0, 1, 3, 2))

                    # flat_bev11 = torch.matmul( flat_bev, self.param_point1[i] )
                    # # flat_bev11 = torch.matmul( self.param_point1_m[i], flat_bev11)
                    # flat_bev = self.fbn1[i](flat_bev11)
                    # flat_bev = torch.matmul(self.param_point_1[i], flat_bev)

                    # flat_bev = torch.matmul( flat_bev, self.param_point11[i] )
                    # # flat_bev = torch.matmul( self.param_point11_m[i], flat_bev)
                    # flat_bev = self.fbn11[i](flat_bev)
                    # flat_bev = torch.matmul(self.param_point_11[i], flat_bev)

                    # flat_bev = torch.matmul( flat_bev, self.param_point111[i] )
                    # # flat_bev = torch.matmul( self.param_point111_m[i], flat_bev)
                    # flat_bev = self.fbn111[i](flat_bev) + flat_bev11
                    # flat_bev = torch.matmul(self.param_point_111[i], flat_bev)
                    # flat_bev = flat_bev.permute((0, 1, 3, 2))

                    # bev_img_fp = self.bev_quant(flat_bev)
                    # collect.append(bev_img_fp)
                collect__ = self.bnMergeFD_col[i](torch.concat([flat_bev_ret0, flat_bev_ret1, flat_bev_ret2], dim = 1))
                flat_bev_col.append(collect__)
            return self.bnMergeALL(torch.concat(flat_bev_col, dim = 1))
        trans_feat = self.bnMergeALL(torch.concat(flat_bev_col, dim = 1))
        return trans_feat

    def fuse_model(self):
        """Perform model fusion on the modules."""
        self.depth_net.fuse_model()
        self.feat_net.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.dquant_stub.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "saturate": True},
            activation_calibration_qkwargs={"dtype": qint16, "saturate": True},
        )
        # 加入BEV特征、深度量化配置，统一qint16
        self.bev_quant.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "saturate": True},
            activation_calibration_qkwargs={"dtype": qint16, "saturate": True},
        )
        self.depth_bev_quant.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "saturate": True},
            activation_calibration_qkwargs={"dtype": qint16, "saturate": True},
        )
        super().set_qconfig()
