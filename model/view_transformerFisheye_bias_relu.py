# Copyright (c) Horizon Robotics. All rights reserved.
###########################
# matmul version

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

class ViewTransformerFisheye(nn.Module):
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
        useSpherical = True,
        feat_hw = [-1, -1],
        original_gridsample = False,
        feat_channels = -1,
        feat_scale = (1/16, 1/16),
        svd_rank = 16,
        num_tile = 4,
        learning_point = True,
        learning_point_version = '3',
        hidden_matmul = 256,
    ):
        super(ViewTransformerFisheye, self).__init__()
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

        self.num_layer = 3

        print("learning_point: hidden_matmul: ", self.learning_point_version, hidden_matmul, self.grid_size[0])
        if learning_point:
            if self.learning_point_version == '9' or self.learning_point_version == '10' or \
                self.learning_point_version == '11':
                self.num_layer = 3
                data1 = torch.rand((num_views, feat_hw[0], hidden_matmul)) * 2 - 1
                data11 = torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1
                data111 = torch.rand((num_views, hidden_matmul, self.grid_size[0])) * 2 - 1
                data_111_final = torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1
                data2 = torch.rand((num_views, feat_hw[1], hidden_matmul)) * 2 - 1
                data22 = torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1
                data222 = torch.rand((num_views, hidden_matmul, self.grid_size[1])) * 2 - 1
                data_1 = torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1
                data_11 = torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1
                data_111 = torch.rand((num_views, self.grid_size[0], self.grid_size[1])) * 2 - 1
                data_2 = torch.rand((num_views, feat_hw[0], feat_hw[0])) * 2 - 1
                data_22 = torch.rand((num_views, feat_hw[0], feat_hw[0])) * 2 - 1
                data_222 = torch.rand((num_views, feat_hw[0], feat_hw[0])) * 2 - 1
                row_sum1 = feat_hw[0] * self.num_views
                row_sum11 = hidden_matmul * self.num_views
                row_sum111 = hidden_matmul * self.num_views
                row_sum2 = feat_hw[1] * self.num_views
                row_sum22 = hidden_matmul * self.num_views
                row_sum222 = hidden_matmul * self.num_views
                row_sum_1 = self.grid_size[1] #* self.num_views
                row_sum_11 = self.grid_size[1] #* self.num_views
                row_sum_111 = self.grid_size[1] #* self.num_views
                row_sum_2 = feat_hw[0] #* self.num_views
                row_sum_22 = feat_hw[0] #* self.num_views
                row_sum_222 = feat_hw[0] #* self.num_views
                self.param_point1 = nn.Parameter(data = data1 / row_sum1, requires_grad = True)
                self.param_point11 = nn.Parameter(data = data11 / row_sum11, requires_grad = True)
                self.param_point111 = nn.Parameter(data = data111 / row_sum111, requires_grad = True)
                self.param_point2 = nn.Parameter(data = data2 / row_sum2, requires_grad = True)
                self.param_point22 = nn.Parameter(data = data22 / row_sum22, requires_grad = True)
                self.param_point222 = nn.Parameter(data = data222 / row_sum222, requires_grad = True)
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

                self.param_point_1 = nn.Parameter(data = data_1 / row_sum_1, requires_grad = True)
                self.param_point_11 = nn.Parameter(data = data_11 / row_sum_11, requires_grad = True)
                self.param_point_111 = nn.Parameter(data = data_111 / row_sum_111, requires_grad = True)
                
                self.param_point_2 = nn.Parameter(data = data_2 / row_sum_2, requires_grad = True)
                self.param_point_22 = nn.Parameter(data = data_22 / row_sum_22, requires_grad = True)
                self.param_point_222 = nn.Parameter(data = data_222 / row_sum_222, requires_grad = True)
                self.param_point_111_final = nn.Parameter(data = data_111_final / row_sum_111, requires_grad = True)
            elif self.learning_point_version == '3':
                data1 = torch.rand((num_views, feat_hw[0], hidden_matmul)) * 2 - 1
                data11 = torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1
                data111 = torch.rand((num_views, hidden_matmul, self.grid_size[0])) * 2 - 1
                data2 = torch.rand((num_views, feat_hw[1], hidden_matmul)) * 2 - 1
                data22 = torch.rand((num_views, hidden_matmul, hidden_matmul)) * 2 - 1
                data222 = torch.rand((num_views, hidden_matmul, self.grid_size[1])) * 2 - 1

                # data1 = torch.randn((num_views, feat_hw[0], self.grid_size[0]))
                # data2 = torch.randn((num_views, feat_hw[1], self.grid_size[1]))
                # data1 = data1 / (data1.max() + data1.min().abs())
                # data2 = data2 / (data2.max() + data2.min().abs())

                # row_sum1 = torch.sum(data1, dim = (2), keepdim = True)
                # row_sum1[row_sum1 == 0] = 1
                # row_sum2 = torch.sum(data2, dim = (2), keepdim = True)
                # row_sum2[row_sum2 == 0] = 1

                row_sum1 = feat_hw[0] * self.num_views
                row_sum11 = hidden_matmul * self.num_views
                row_sum111 = hidden_matmul * self.num_views
                row_sum2 = feat_hw[1] * self.num_views
                row_sum22 = hidden_matmul * self.num_views
                row_sum222 = hidden_matmul * self.num_views
                self.param_point1 = nn.Parameter(data = data1 / row_sum1, requires_grad = True)
                self.param_point11 = nn.Parameter(data = data11 / row_sum11, requires_grad = True)
                self.param_point111 = nn.Parameter(data = data111 / row_sum111, requires_grad = True)
                self.param_point2 = nn.Parameter(data = data2 / row_sum2, requires_grad = True)
                self.param_point22 = nn.Parameter(data = data22 / row_sum22, requires_grad = True)
                self.param_point222 = nn.Parameter(data = data222 / row_sum222, requires_grad = True)


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

                # drow_sum1 = torch.sum(ddata1, dim = (2), keepdim = True)
                # drow_sum1[drow_sum1 == 0] = 1
                # drow_sum2 = torch.sum(ddata2, dim = (2), keepdim = True)
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
                ddata2 = torch.rand((num_views, feat_hw[1], self.grid_size[0])) * 2 - 1

                drow_sum1 = 1 #feat_hw[0] * self.num_views
                drow_sum2 = 1 #feat_hw[1] * self.num_views
                self.dparam_point1 = nn.Parameter(data = ddata1 / drow_sum1, requires_grad = True)
                self.dparam_point2 = nn.Parameter(data = ddata2 / drow_sum2, requires_grad = True)
        # if learning_point:
        #     self.spatial_decay_scale = 2.0
        #     self.use_gaussian_init = True

        #     Hf, Wf = self.feat_hw
        #     Hbev, Wbev = self.grid_size  # 固定128,128

        #     # 1. 图像像素网格 [-1,1]
        #     y_img = torch.linspace(-1., 1., Hf).unsqueeze(1)  # [Hf,1]
        #     x_img = torch.linspace(-1., 1., Wf).unsqueeze(0)  # [1,Wf]
        #     pixel_radius = torch.sqrt(x_img ** 2 + y_img ** 2)
        #     decay_mask = torch.exp(-self.spatial_decay_scale * pixel_radius)
        #     decay_coeff = decay_mask.mean()

        #     # BEV网格中心 [-1,1]
        #     bev_y = torch.linspace(-1., 1., Hbev).unsqueeze(0)  # [1, Hbev]
        #     bev_x = torch.linspace(-1., 1., Wbev).unsqueeze(0)  # [1, Wbev]

        #     shape_p1 = (self.num_views, Hf, Hbev)
        #     shape_p2 = (self.num_views, Wf, Wbev)

        #     if self.use_gaussian_init:
        #         # ========== H维度映射矩阵 [Hf, Hbev]：每个BEV y对应独立高斯中心 ==========
        #         h_axis = torch.linspace(-2.0, 2.0, Hf).unsqueeze(-1)  # [Hf, 1]
        #         sigma_h = max(0.6, Hf / 32.0)
        #         # 广播：[Hf,1] - [1,Hbev] → [Hf, Hbev]
        #         gauss_h = torch.exp(-(h_axis - bev_y).pow(2) / (2 * sigma_h ** 2))

        #         # ========== W维度映射矩阵 [Wf, Wbev] ==========
        #         w_axis = torch.linspace(-2.0, 2.0, Wf).unsqueeze(-1)  # [Wf,1]
        #         sigma_w = max(0.8, Wf / 32.0)
        #         gauss_w = torch.exp(-(w_axis - bev_x).pow(2) / (2 * sigma_w ** 2))
        #         gauss_w = gauss_w * decay_coeff

        #         # 批量扩展多视图
        #         data1 = gauss_h.unsqueeze(0).repeat(self.num_views, 1, 1)
        #         data2 = gauss_w.unsqueeze(0).repeat(self.num_views, 1, 1)
        #         ddata1 = gauss_h.unsqueeze(0).repeat(self.num_views, 1, 1)
        #         ddata2 = gauss_w.unsqueeze(0).repeat(self.num_views, 1, 1)

        #     else:
        #         decay_coeff = decay_mask.mean()
        #         data1 = (torch.rand(shape_p1) * 2 - 1) * decay_coeff
        #         data2 = (torch.rand(shape_p2) * 2 - 1) * decay_coeff
        #         ddata1 = (torch.rand(shape_p1) * 2 - 1) * decay_coeff
        #         ddata2 = (torch.rand(shape_p2) * 2 - 1) * decay_coeff

        #     # 行归一：每行权重和为1，不abs保留正负
        #     def row_normalize(mat: torch.Tensor):
        #         row_sum = mat.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        #         return mat / row_sum

        #     data1 = row_normalize(data1)
        #     data2 = row_normalize(data2)
        #     ddata1 = row_normalize(ddata1)
        #     ddata2 = row_normalize(ddata2)

        #     self.param_point1 = nn.Parameter(data=data1, requires_grad=True)
        #     self.param_point2 = nn.Parameter(data=data2, requires_grad=True)
        #     self.dparam_point1 = nn.Parameter(data=ddata1, requires_grad=True)
        #     self.dparam_point2 = nn.Parameter(data=ddata2, requires_grad=True)

        # ===== Q = PAH + B: output bias (bev_bias) =====
        self.use_bev_bias = True

        # ===== chain ReLU: real depth for v1-v3 (False = reproduce old linear chains) =====
        self.use_chain_relu = True
        if self.use_bev_bias:
            # per-view BEV spatial bias, shared across channels: num_views x 1 x H_b x W_b
            self.bev_bias = nn.Parameter(
                data = torch.zeros(self.num_views, 1, self.grid_size[0], self.grid_size[1]),
                requires_grad = True,
            )
            # per-channel variant (closer to "any same-shape matrix", x feat_channels params):
            # self.bev_bias = nn.Parameter(
            #     data = torch.zeros(self.num_views, self.feat_channels, self.grid_size[0], self.grid_size[1]),
            #     requires_grad = True,
            # )

        self.ref_point = None

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
            return self._spatial_transfom_learn(feats), None
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

@OBJECT_REGISTRY.register
class LSSTransformerFisheye(ViewTransformerFisheye):
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
        outch = feat_channels
        if self.learning_point_version == '9':
            outch = feat_channels * self.num_layer
        self.feat_net = ConvModule2d(
            in_channels=in_channels,
            out_channels=outch,
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
        # self.omni_ocam = ocamCameraTorch(filename=self.ocam_path, fov=self.ocam_fov)
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
        self.bnMergeFD_col = nn.ModuleList([ConvModule2d(self.feat_channels * self.num_layer, self.feat_channels, (1, 1),  
                                                stride=1, padding=0, norm_layer=nn.BatchNorm2d(self.depth),
                                                act_layer=nn.ReLU(inplace=True),) for i in range(self.num_views)])
        self.bnMergeALL = nn.Sequential(
                                ConvModule2d(self.feat_channels * self.num_views, self.depth, (3, 3),  
                                                stride=1, padding=1, norm_layer=nn.BatchNorm2d(self.depth),
                                                act_layer=nn.ReLU(inplace=True),),)

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

    def get_sphere_grid(self):
        proj_pts = self.frustum.cpu().numpy()
        sphere_grid = []
        for d in range(proj_pts.shape[0]):
            mapx, mapy = self.omni_ocam.world2cam(
                proj_pts[d, ...].reshape(-1, 3).T)
            mapx, mapy = mapx.reshape(
                self.feature_size), mapy.reshape(self.feature_size)
            mapx, mapy = mapx * 2 / self.omni_ocam.width - \
                1, mapy * 2 / self.omni_ocam.height - 1
            grid = torch.from_numpy(np.stack([mapx, mapy], axis=-1))
            sphere_grid.append(grid)
        sphere_grid = torch.stack(sphere_grid, dim=0)
        depth_coords = torch.zeros(
            size=(self.D, self.feature_size[0], self.feature_size[1], 1))
        sphere_grid = torch.cat(
            (sphere_grid, depth_coords), dim=-1)  # [D, H, W, 3]
        return nn.Parameter(sphere_grid, requires_grad=False)

    def _gen_reference_point(
        self, homography: Tensor, feat_hw: Tuple[int, int]
    ) -> Tuple[Tensor]:
        """Generate reference points using homography matrix and feature tensor.

        Args:
            homography: Homography matrix.
            feat_hw: View transformer input shape
                     for generationg reference points.

        Returns:
            Tuple containing the generated feature points and depth points.
        """
        if self.useSpherical:
            coords = self._gen_3d_points( # torch.Size([128, 128, 64, 4])
                self.z_range, cal_minmax = self.cal_minmax
            ).to( device = homography.device )
        else:
            coords = self._gen_3d_points_origin( # torch.Size([128, 128, 64, 4])
                self.z_range, cal_minmax=self.cal_minmax
            ).to(device=homography.device)
        H, W, Z = coords.shape[:3]

        # mxx = torch.max(coords[..., 0])
        # mix = torch.min(coords[..., 0])
        # mxy = torch.max(coords[..., 1])
        # miy = torch.min(coords[..., 1])
        # mxz = torch.max(coords[..., 2])
        # miz = torch.min(coords[..., 2])

        if self.useLidar2cam:
            new_coords = []
            # (M * coord)^T=(coord^T * M^T)
            for homo in homography:
                new_coord = torch.matmul(coords, homo.permute((1, 0))).float()
                new_coord = new_coord.permute((2, 0, 1, 3)) # torch.Size([64, 128, 128, 4])
                new_coords.append(new_coord)
            new_coords = torch.stack(new_coords, dim=1) # torch.Size([64, 3 * 4, 128, 128, 4])
        else:
            new_coords = coords.repeat(len(homography), 1, 1, 1, 1).permute(3, 0, 1, 2, 4).float()
        
        # mapx, mapy = self.omni_ocam.world2cam(new_coords[..., :3].reshape(-1, 3).cpu().numpy().T)

        # mxxi = torch.max(new_coords[..., 0])
        # mixi = torch.min(new_coords[..., 0])
        # mxyi = torch.max(new_coords[..., 1])
        # miyi = torch.min(new_coords[..., 1])
        # mxzi = torch.max(new_coords[..., 2])
        # mizi = torch.min(new_coords[..., 2])

        mapxT, mapyT = self.omni_ocam.world2camTorch(new_coords[..., :3].reshape(-1, 3).T)

        # mxx_ = torch.max(mapxT)
        # mix_ = torch.min(mapxT)
        # mxy_ = torch.max(mapyT)
        # miy_ = torch.min(mapyT)

        # orig_hw = meta["img"][0].shape[1:]
        # scales = (feat_hw[0] / orig_hw[0], feat_hw[1] / orig_hw[1])
        mapxT = mapxT * self.feat_scale[0]
        mapyT = mapyT * self.feat_scale[1]

        # mxxs_ = torch.max(mapxT)
        # mixs_ = torch.min(mapxT)
        # mxys_ = torch.max(mapyT)
        # miys_ = torch.min(mapyT)

        # mapx = torch.from_numpy(mapx).to(mapxT.device) #############################################
        # mapy = torch.from_numpy(mapy).to(mapxT.device)
        # t1 = mapx[mapx!=-1]
        # t2 = mapy[mapy!=-1]
        # t11 = mapxT[mapxT!=-scales[0]]
        # t22 = mapyT[mapyT!=-scales[1]]
        # i0 = (t1 == t11).all()
        # i1 = (t2 == t22).all()
        # p0 = (mapx == mapxT).all()
        # p1 = (mapy == mapyT).all()
        # n0 = mapx[mapx != mapxT]
        # n00 = mapxT[mapx != mapxT]
        # n1 = mapy[mapy != mapyT]
        # n11 = mapyT[mapy != mapyT]
        # k1 = torch.max(torch.abs(mapx - mapxT)) #.all()
        # k2 = torch.max(torch.abs(mapy - mapyT)) #.all()) #############################################

        mapxT = mapxT.reshape(new_coords.shape[:-1]).unsqueeze(-1)
        mapyT = mapyT.reshape(new_coords.shape[:-1]).unsqueeze(-1)
        new_coords = torch.concat([mapxT, mapyT, new_coords[..., 2:]], dim = -1)

        B = new_coords.shape[1] // self.num_views

        new_coords = (
            new_coords.view(-1, B, self.num_views, H, W, 4)
            .permute(0, 2, 1, 3, 4, 5)
            .contiguous()
        )

        # d = torch.clamp(new_coords[..., 2], min=0.05)
        # X = (new_coords[..., 0] / d).long()
        # Y = (new_coords[..., 1] / d).long()
        X = (new_coords[..., 0]).long()
        Y = (new_coords[..., 1]).long()
        D = new_coords[..., 2].long()

        feat_h, feat_w = feat_hw
        #  X([64, 4, 1, 128, 128]) Y([64, 4, 1, 128, 128]) grid_size[128, 128] feat_h 50 feat_w 50
        # valid = (X >= 0) & (X < feat_w) & (Y >= 0) & (Y < feat_h) 

        N_bev = self.grid_size[0] * self.grid_size[1]
        N_img = feat_h * feat_w 

        # # 修复：取出全部维度索引，计算BEV一维全局下标
        # idx_all = torch.where(valid)
        # _, _, _, bev_y, bev_x = idx_all
        # bevx = bev_x.cpu().numpy().tolist()
        # valid_bev_idx = bev_y * self.grid_size[1] + bev_x  # 展平bev网格索引

        # if not self.useTopKReferencePoint:
        #     Wmat = torch.zeros((N_bev, N_img), dtype=torch.float32, device=X.device)
        #     valid_pixel_index = Y[valid] * feat_w + X[valid]
        #     # 两个同长度一维索引，批量赋值
        #     Wmat[valid_bev_idx, valid_pixel_index] = 1.0

        #     row_sum = torch.sum(Wmat, dim = (1), keepdim = True)
        #     row_sum[row_sum == 0] = 1
        #     Wmat = Wmat / row_sum

        #     Wmat = Wmat.to(device = homography.device)

        #     U, S, Vh = torch.linalg.svd(Wmat, full_matrices=False)
        #     S_rank = torch.diag(S)[:, :self.svd_rank]
        #     U_rank = torch.matmul(U, S_rank).unsqueeze(0)
        #     Vh_rank = Vh[:self.svd_rank, :].unsqueeze(0)

        #     return (U_rank, Vh_rank), (U_rank, Vh_rank)

        idx = (
            (
                torch.linspace(0, self.num_views - 1, self.num_views)
                .reshape((1, self.num_views, 1, 1, 1))
                .repeat(Z, 1, B, H, W)
            )
            .long()
            .to(device=homography.device)
        )

        new_coords = torch.stack([X, Y, D, idx], dim=-1)

        # mxxds_ = torch.max(X)
        # mixds_ = torch.min(X)
        # mxyds_ = torch.max(Y)
        # miyds_ = torch.min(Y)
        # mxzds_ = torch.max(D)
        # mizds_ = torch.min(D)

        invalid = (
            (new_coords[..., 0] < 0)
            | (new_coords[..., 0] >= feat_w)
            | (new_coords[..., 1] < 0)
            | (new_coords[..., 1] >= feat_h)
            | (new_coords[..., 2] < 0)
            | (new_coords[..., 2] >= self.depth)
        )
        # kk = new_coords[~invalid] #[..., 2]
        # minxx = kk[..., 0].float().min().item()
        # maxxx = kk[..., 0].float().max().item()
        # minyy = kk[..., 1].float().min().item()
        # maxyy = kk[..., 1].float().max().item()
        # minzz = kk[..., 2].float().min().item()
        # maxzz = kk[..., 2].float().max().item()
        # kk = kk.float().mean()

        if self.use_vtv2 is True:
            new_coords[invalid] = torch.tensor(
                65535,
            ).to(device=homography.device)
            new_coords = new_coords.view(-1, B, H, W, 4)
            coords_xy = new_coords[..., :2].to(dtype=torch.float)
            coords_xy[..., 0] = coords_xy[..., 0] / feat_w
            coords_xy[..., 1] = coords_xy[..., 1] / feat_h
            center = (
                torch.tensor(
                    (0.5, 0),
                )
                .to(device=homography.device)
                .view(1, -1)
            )
            dist = torch.cdist(coords_xy, center, p=2).squeeze(-1)

            dist_sorted, indices_sorted = torch.sort(dist, dim=0)
            indices_sorted = indices_sorted.unsqueeze(-1).expand(
                -1, -1, -1, -1, 4
            )
            coords_sorted = torch.gather(
                new_coords, dim=0, index=indices_sorted
            )
            mask = torch.zeros_like(dist_sorted, dtype=torch.bool)

            mask[1:] = dist_sorted[1:] == dist_sorted[:-1]
            dist_cleaned = dist_sorted.masked_fill(mask, 65535)
            mask = mask.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
            coords_cleaned = coords_sorted.masked_fill(mask, 65535)
            _, indices = dist_cleaned.topk(
                self.num_points, dim=0, largest=False
            )
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
            topk_coords = torch.gather(coords_cleaned, dim=0, index=indices)
            X = topk_coords[..., 0]
            Y = topk_coords[..., 1]
            D = topk_coords[..., 2]
            idx = topk_coords[..., 3]
        else:
            new_coords[invalid] = torch.tensor(
                (feat_w - 1, feat_h - 1, self.depth, self.num_views - 1)
            ).to(device=homography.device)
            new_coords = new_coords.view(-1, B, H, W, 4)
            rank = (
                new_coords[..., 2] * feat_h * feat_w * self.num_views
                + new_coords[..., 1] * feat_w * self.num_views
                + new_coords[..., 0] * self.num_views
                + new_coords[..., 3]
            )
            rank, _ = rank.topk(self.num_points, dim=0, largest=False)
            D = rank // (feat_h * feat_w * self.num_views)
            rank = rank % (feat_h * feat_w * self.num_views)

            Y = rank // (feat_w * self.num_views)
            rank = rank % (feat_w * self.num_views)

            X = rank // self.num_views
            idx = rank % self.num_views

        idx_Y = idx * feat_h + Y
        feat_coords = torch.stack((X, idx_Y), dim=-1)
        feat_points = adjust_coords(feat_coords, self.grid_size)

        X_Y = Y * feat_w + X
        idx_D = idx * self.depth + D
        depth_coords = torch.stack((X_Y, idx_D), dim=-1)
        depth_points = adjust_coords(depth_coords, self.grid_size)
        feat_points = feat_points.view(-1, H, W, 2)
        depth_points = depth_points.view(-1, H, W, 2)

        # quant_stub, range to [ -1, 1 ]
        feat_points = self.nofloat_warp_gridsample_operator(feat_points,
                                                            self.num_views * feat_h,
                                                            feat_w) # torch.Size([50, 128, 128, 2])
        # feat = feat.view(B, C, -1, W) # self.num_views * H, W torch.Size([1, 64, 200, 50])  torch.Size([10, 128, 128, 2])
        depth_points = self.nofloat_warp_gridsample_operator(depth_points,
                                                             self.depth * self.num_views,
                                                             feat_h * feat_w) # torch.Size([50, 128, 128, 2])
        fU, fV, dU, dV = [], [], [], []
        for i in range(self.num_points):
            Wmat = torch.zeros((N_bev, N_img), dtype=torch.float32)

            x = feat_points[i][..., 0]
            y = feat_points[i][..., 1]

            x = 0.5 * (x + 1) * (feat_w - 1)
            y = 0.5 * (y + 1) * (feat_h - 1)

            x = torch.round(x).int()
            y = torch.round(y).int()

            x = torch.clamp(x, 0, feat_w - 1)
            y = torch.clamp(y, 0, feat_h - 1)

            valid_pixel_index = y * feat_w + x   # 256 * 2500 too large
            valid_pixel_index = valid_pixel_index.flatten()

            Wmat[torch.arange(N_bev).int(), valid_pixel_index] = 1.0

            row_sum = torch.sum(Wmat, dim = (1), keepdim = True)
            row_sum[row_sum == 0] = 1
            Wmat = Wmat / row_sum

            Wmat = Wmat.to(device = homography.device)

            U, S, Vh = torch.linalg.svd(Wmat, full_matrices=False)
            S_rank = torch.diag(S)[:, :self.svd_rank]
            U_rank = torch.matmul(U, S_rank)
            Vh_rank = Vh[:self.svd_rank, :]

            fU.append(U_rank)
            fV.append(Vh_rank)

        fU_out = torch.stack(fU, dim = 0)
        fV_out = torch.stack(fV, dim=0)
        for i in range(self.num_points):
            Wmat = torch.zeros((N_bev, N_img), dtype=torch.float32)

            x = depth_points[i][..., 0]
            y = depth_points[i][..., 1]

            x = 0.5 * (x + 1) * (feat_w - 1)
            y = 0.5 * (y + 1) * (feat_h - 1)

            x = torch.round(x).int()
            y = torch.round(y).int()

            x = torch.clamp(x, 0, feat_w - 1)
            y = torch.clamp(y, 0, feat_h - 1)

            valid_pixel_index = y * feat_w + x   # 256 * 2500 too large
            valid_pixel_index = valid_pixel_index.flatten()

            Wmat[torch.arange(N_bev).int(), valid_pixel_index] = 1.0

            row_sum = torch.sum(Wmat, dim = (1), keepdim = True)
            row_sum[row_sum == 0] = 1
            Wmat = Wmat / row_sum

            Wmat = Wmat.to(device = homography.device)

            U, S, Vh = torch.linalg.svd(Wmat, full_matrices=False)
            S_rank = torch.diag(S)[:, :self.svd_rank]
            dU_rank = torch.matmul(U, S_rank)
            dVh_rank = Vh[:self.svd_rank, :]

            dU.append(dU_rank)
            dV.append(dVh_rank)

        dU_out = torch.stack(dU, dim = 0)
        dV_out = torch.stack(dV, dim=0)

        return (fU_out, fV_out), (dU_out, dV_out)
        # dfeat = dfeat.view(B, 1, -1, H * W) # self.depth * self.num_views, H * W   torch.Size([1, 1, 256, 2500])  torch.Size([10, 128, 128, 2])
        # [-1, 1] fix to integer

        # mian0 = feat_points.min().item()
        # mxan0 = feat_points.max().item()
        # mian1 = feat_points.min().item()
        # mxan1 = feat_points.max().item()
        # dmian0 = depth_points.min().item()
        # dmxan0 = depth_points.max().item()
        # dmian1 = depth_points.min().item()
        # dmxan1 = depth_points.max().item()

        # feat_points = self.grid_sample.fix_grid_integer_origin_label1(feat_points, self.feat_channels, self.num_views * feat_h, feat_w)  # torch.Size([50, 64, 16384])
        # depth_points = self.grid_sample.fix_grid_integer_origin_label1(depth_points, 1, self.depth * self.num_views, feat_h * feat_w)  # torch.Size([50, 1, 16384])
        return (feat_points, depth_points)
    
    def nofloat_warp_gridsample_operator(self, grid, feat_height2, feat_width3):
        # range to [ -1, 1 ]
        grid = grid.float()

        # convert grid format from 'delta' to 'norm'
        n = grid.size(0)
        h = grid.size(1)
        w = grid.size(2)
        base_coord_y = (
            torch.arange(h, dtype=grid.dtype, device=grid.device)
            .unsqueeze(-1)
            .unsqueeze(0)
            .expand(n, h, w)
        )
        base_coord_x = (
            torch.arange(w, dtype=grid.dtype, device=grid.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(n, h, w)
        )
        absolute_grid_x = grid[:, :, :, 0] + base_coord_x
        absolute_grid_y = grid[:, :, :, 1] + base_coord_y
        norm_grid_x = absolute_grid_x * 2 / (feat_width3 - 1) - 1
        norm_grid_y = absolute_grid_y * 2 / (feat_height2 - 1) - 1
        norm_grid = torch.stack((norm_grid_x, norm_grid_y), dim=-1) # [-1, 1], gridsample range
        return norm_grid

    def _gen_reference_point_origin(
        self, homography: Tensor, feat_hw: Tuple[int, int]
    ) -> Tuple[Tensor]:
        """Generate reference points using homography matrix and feature tensor.

        Args:
            homography: Homography matrix.
            feat_hw: View transformer input shape
                     for generationg reference points.

        Returns:
            Tuple containing the generated feature points and depth points.
        """
        if self.useSpherical:
            coords = self._gen_3d_points( # torch.Size([128, 128, 64, 4])
                self.z_range, cal_minmax = self.cal_minmax
            ).to( device = homography.device )
        else:
            coords = self._gen_3d_points_origin( # torch.Size([128, 128, 64, 4])
                self.z_range, cal_minmax=self.cal_minmax
            ).to(device=homography.device)
        H, W, Z = coords.shape[:3]

        # mxx = torch.max(coords[..., 0])
        # mix = torch.min(coords[..., 0])
        # mxy = torch.max(coords[..., 1])
        # miy = torch.min(coords[..., 1])
        # mxz = torch.max(coords[..., 2])
        # miz = torch.min(coords[..., 2])

        if self.useLidar2cam:
            new_coords = []
            # (M * coord)^T=(coord^T * M^T)
            for homo in homography:
                new_coord = torch.matmul(coords, homo.permute((1, 0))).float()
                new_coord = new_coord.permute((2, 0, 1, 3)) # torch.Size([64, 128, 128, 4])
                new_coords.append(new_coord)
            new_coords = torch.stack(new_coords, dim=1) # torch.Size([64, 3 * 4, 128, 128, 4])
        else:
            new_coords = coords.repeat(len(homography), 1, 1, 1, 1).permute(3, 0, 1, 2, 4).float()
        
        # mapx, mapy = self.omni_ocam.world2cam(new_coords[..., :3].reshape(-1, 3).cpu().numpy().T)

        # mxxi = torch.max(new_coords[..., 0])
        # mixi = torch.min(new_coords[..., 0])
        # mxyi = torch.max(new_coords[..., 1])
        # miyi = torch.min(new_coords[..., 1])
        # mxzi = torch.max(new_coords[..., 2])
        # mizi = torch.min(new_coords[..., 2])

        mapxT, mapyT = self.omni_ocam.world2camTorch(new_coords[..., :3].reshape(-1, 3).T)

        # mxx_ = torch.max(mapxT)
        # mix_ = torch.min(mapxT)
        # mxy_ = torch.max(mapyT)
        # miy_ = torch.min(mapyT)

        # orig_hw = meta["img"][0].shape[1:]
        # scales = (feat_hw[0] / orig_hw[0], feat_hw[1] / orig_hw[1])
        mapxT = mapxT * self.feat_scale[0]
        mapyT = mapyT * self.feat_scale[1]

        # mxxs_ = torch.max(mapxT)
        # mixs_ = torch.min(mapxT)
        # mxys_ = torch.max(mapyT)
        # miys_ = torch.min(mapyT)

        # mapx = torch.from_numpy(mapx).to(mapxT.device) #############################################
        # mapy = torch.from_numpy(mapy).to(mapxT.device)
        # t1 = mapx[mapx!=-1]
        # t2 = mapy[mapy!=-1]
        # t11 = mapxT[mapxT!=-scales[0]]
        # t22 = mapyT[mapyT!=-scales[1]]
        # i0 = (t1 == t11).all()
        # i1 = (t2 == t22).all()
        # p0 = (mapx == mapxT).all()
        # p1 = (mapy == mapyT).all()
        # n0 = mapx[mapx != mapxT]
        # n00 = mapxT[mapx != mapxT]
        # n1 = mapy[mapy != mapyT]
        # n11 = mapyT[mapy != mapyT]
        # k1 = torch.max(torch.abs(mapx - mapxT)) #.all()
        # k2 = torch.max(torch.abs(mapy - mapyT)) #.all()) #############################################

        mapxT = mapxT.reshape(new_coords.shape[:-1]).unsqueeze(-1)
        mapyT = mapyT.reshape(new_coords.shape[:-1]).unsqueeze(-1)
        new_coords = torch.concat([mapxT, mapyT, new_coords[..., 2:]], dim = -1)

        B = new_coords.shape[1] // self.num_views

        new_coords = (
            new_coords.view(-1, B, self.num_views, H, W, 4)
            .permute(0, 2, 1, 3, 4, 5)
            .contiguous()
        )

        # d = torch.clamp(new_coords[..., 2], min=0.05)
        # X = (new_coords[..., 0] / d).long()
        # Y = (new_coords[..., 1] / d).long()
        X = (new_coords[..., 0]).long()
        Y = (new_coords[..., 1]).long()
        D = new_coords[..., 2].long()

        idx = (
            (
                torch.linspace(0, self.num_views - 1, self.num_views)
                .reshape((1, self.num_views, 1, 1, 1))
                .repeat(Z, 1, B, H, W)
            )
            .long()
            .to(device=homography.device)
        )
        new_coords = torch.stack([X, Y, D, idx], dim=-1)

        # mxxds_ = torch.max(X)
        # mixds_ = torch.min(X)
        # mxyds_ = torch.max(Y)
        # miyds_ = torch.min(Y)
        # mxzds_ = torch.max(D)
        # mizds_ = torch.min(D)

        feat_h, feat_w = feat_hw
        invalid = (
            (new_coords[..., 0] < 0)
            | (new_coords[..., 0] >= feat_w)
            | (new_coords[..., 1] < 0)
            | (new_coords[..., 1] >= feat_h)
            | (new_coords[..., 2] < 0)
            | (new_coords[..., 2] >= self.depth)
        )

        # kk = new_coords[~invalid] #[..., 2]
        # minxx = kk[..., 0].float().min().item()
        # maxxx = kk[..., 0].float().max().item()
        # minyy = kk[..., 1].float().min().item()
        # maxyy = kk[..., 1].float().max().item()
        # minzz = kk[..., 2].float().min().item()
        # maxzz = kk[..., 2].float().max().item()
        # kk = kk.float().mean()

        if self.use_vtv2 is True:
            new_coords[invalid] = torch.tensor(
                65535,
            ).to(device=homography.device)
            new_coords = new_coords.view(-1, B, H, W, 4)
            coords_xy = new_coords[..., :2].to(dtype=torch.float)
            coords_xy[..., 0] = coords_xy[..., 0] / feat_w
            coords_xy[..., 1] = coords_xy[..., 1] / feat_h
            center = (
                torch.tensor(
                    (0.5, 0),
                )
                .to(device=homography.device)
                .view(1, -1)
            )
            dist = torch.cdist(coords_xy, center, p=2).squeeze(-1)

            dist_sorted, indices_sorted = torch.sort(dist, dim=0)
            indices_sorted = indices_sorted.unsqueeze(-1).expand(
                -1, -1, -1, -1, 4
            )
            coords_sorted = torch.gather(
                new_coords, dim=0, index=indices_sorted
            )
            mask = torch.zeros_like(dist_sorted, dtype=torch.bool)

            mask[1:] = dist_sorted[1:] == dist_sorted[:-1]
            dist_cleaned = dist_sorted.masked_fill(mask, 65535)
            mask = mask.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
            coords_cleaned = coords_sorted.masked_fill(mask, 65535)
            _, indices = dist_cleaned.topk(
                self.num_points, dim=0, largest=False
            )
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
            topk_coords = torch.gather(coords_cleaned, dim=0, index=indices)
            X = topk_coords[..., 0]
            Y = topk_coords[..., 1]
            D = topk_coords[..., 2]
            idx = topk_coords[..., 3]
        else:
            new_coords[invalid] = torch.tensor(
                (feat_w - 1, feat_h - 1, self.depth, self.num_views - 1)
            ).to(device=homography.device)
            new_coords = new_coords.view(-1, B, H, W, 4)
            rank = (
                new_coords[..., 2] * feat_h * feat_w * self.num_views
                + new_coords[..., 1] * feat_w * self.num_views
                + new_coords[..., 0] * self.num_views
                + new_coords[..., 3]
            )
            rank, _ = rank.topk(self.num_points, dim=0, largest=False)
            D = rank // (feat_h * feat_w * self.num_views)
            rank = rank % (feat_h * feat_w * self.num_views)

            Y = rank // (feat_w * self.num_views)
            rank = rank % (feat_w * self.num_views)

            X = rank // self.num_views
            idx = rank % self.num_views

        idx_Y = idx * feat_h + Y
        feat_coords = torch.stack((X, idx_Y), dim=-1)
        feat_points = adjust_coords(feat_coords, self.grid_size)

        X_Y = Y * feat_w + X
        idx_D = idx * self.depth + D
        depth_coords = torch.stack((X_Y, idx_D), dim=-1)
        depth_points = adjust_coords(depth_coords, self.grid_size)
        feat_points = feat_points.view(-1, H, W, 2)
        depth_points = depth_points.view(-1, H, W, 2)
        return (feat_points, depth_points)
    
        # import matplotlib.pyplot as plt
        # import numpy as np
        # data = norm_grid[0, ..., 0].flatten().cpu().numpy()
        # choose = np.linspace(0, len(data)-1, 200).astype(np.int32)
        # data = data[choose]
        # plt.plot(np.arange(len(data)), data)
        # plt.savefig(f'/home/Desktop/fisheyedod/horizon/script/poolbev/x_{numnum}.jpg')
        # plt.close('all')
        # data = norm_grid[0, ..., 1].flatten().cpu().numpy()
        # data = data[choose]
        # plt.plot(np.arange(len(data)), data)
        # plt.savefig(f'/home/Desktop/fisheyedod/horizon/script/poolbev/y_{numnum}.jpg')
        # plt.close('all')
        # numnum += 1

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

        depth = self.softmax(self.depth_net(feats))

        new_feats = self.feat_net(feats)
        return new_feats, depth

    def _spatial_transfom_learn(self, feats: Tensor) -> Tensor:
        """Apply spatial transformation to the features using the given points.

        Args:
            feats: Tuple of feature tensor and depth tensor.
            points: Tuple of feature points and depth points.

        Returns:
            The transformed feature tensor.
        """
        feat, dfeat = feats # torch.Size([8, 64, 50, 50])      torch.Size([8, 64, 50, 50])

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

        flat_bev_col = []
        if self.learning_point_version == '3':
            for i in range(self.num_views):
                img_feat = feat[i]
                depth_feat = dfeat[i]

                flat_bev = torch.matmul( img_feat, self.param_point2[i] )
                if self.use_chain_relu:
                    flat_bev = torch.relu(flat_bev)
                flat_bev = torch.matmul( flat_bev, self.param_point22[i] )
                if self.use_chain_relu:
                    flat_bev = torch.relu(flat_bev)
                flat_bev = torch.matmul( flat_bev, self.param_point222[i] )
                if self.use_chain_relu:
                    flat_bev = torch.relu(flat_bev)
                if self.training or B > 1:
                    flat_bev = flat_bev.permute((0, 1, 3, 2))
                else:
                    flat_bev = flat_bev.permute((0, 2, 1))
                flat_bev = torch.matmul( flat_bev, self.param_point1[i] )
                if self.use_chain_relu:
                    flat_bev = torch.relu(flat_bev)
                flat_bev = torch.matmul( flat_bev, self.param_point11[i] )
                if self.use_chain_relu:
                    flat_bev = torch.relu(flat_bev)
                flat_bev = torch.matmul( flat_bev, self.param_point111[i] )
                if self.training or B > 1:
                    flat_bev = flat_bev.permute((0, 1, 3, 2))
                else:
                    flat_bev = flat_bev.permute((0, 2, 1))
                    

                flat_depth_bev = torch.matmul( depth_feat, self.dparam_point2[i] )
                if self.use_chain_relu:
                    flat_depth_bev = torch.relu(flat_depth_bev)
                flat_depth_bev = torch.matmul( flat_depth_bev, self.dparam_point22[i] )
                if self.use_chain_relu:
                    flat_depth_bev = torch.relu(flat_depth_bev)
                flat_depth_bev = torch.matmul( flat_depth_bev, self.dparam_point222[i] )
                if self.use_chain_relu:
                    flat_depth_bev = torch.relu(flat_depth_bev)
                if self.training or B > 1:
                    flat_depth_bev = flat_depth_bev.permute((0, 1, 3, 2))
                else:
                    flat_depth_bev = flat_depth_bev.permute((0, 2, 1))
                flat_depth_bev = torch.matmul( flat_depth_bev, self.dparam_point1[i] )
                if self.use_chain_relu:
                    flat_depth_bev = torch.relu(flat_depth_bev)
                flat_depth_bev = torch.matmul( flat_depth_bev, self.dparam_point11[i] )
                if self.use_chain_relu:
                    flat_depth_bev = torch.relu(flat_depth_bev)
                flat_depth_bev = torch.matmul( flat_depth_bev, self.dparam_point111[i] )
                if self.training or B > 1:
                    flat_depth_bev = flat_depth_bev.permute((0, 1, 3, 2))
                else:
                    flat_depth_bev = flat_depth_bev.permute((0, 2, 1))

                # flat_bev_col.append(flat_bev + flat_depth_bev)
                bev_img_fp = self.bev_quant(flat_bev)
                bev_depth_fp = self.depth_bev_quant(flat_depth_bev)
                fused = bev_img_fp * bev_depth_fp
                if self.use_bev_bias:
                    fused = fused + self.bev_bias[i]
                flat_bev_col.append(fused)
        elif self.learning_point_version == '2':
            for i in range(self.num_views):
                img_feat = feat[i]
                depth_feat = dfeat[i]

                flat_bev = torch.matmul( img_feat, self.param_point2[i] )
                if self.use_chain_relu:
                    flat_bev = torch.relu(flat_bev)
                flat_bev = torch.matmul( flat_bev, self.param_point22[i] )
                if self.use_chain_relu:
                    flat_bev = torch.relu(flat_bev)
                if self.training or B > 1:
                    flat_bev = flat_bev.permute((0, 1, 3, 2))
                else:
                    flat_bev = flat_bev.permute((0, 2, 1))
                flat_bev = torch.matmul( flat_bev, self.param_point1[i] )
                if self.use_chain_relu:
                    flat_bev = torch.relu(flat_bev)
                flat_bev = torch.matmul( flat_bev, self.param_point11[i] )
                if self.training or B > 1:
                    flat_bev = flat_bev.permute((0, 1, 3, 2))
                else:
                    flat_bev = flat_bev.permute((0, 2, 1))
                    

                flat_depth_bev = torch.matmul( depth_feat, self.dparam_point2[i] )
                if self.use_chain_relu:
                    flat_depth_bev = torch.relu(flat_depth_bev)
                flat_depth_bev = torch.matmul( flat_depth_bev, self.dparam_point22[i] )
                if self.use_chain_relu:
                    flat_depth_bev = torch.relu(flat_depth_bev)
                if self.training or B > 1:
                    flat_depth_bev = flat_depth_bev.permute((0, 1, 3, 2))
                else:
                    flat_depth_bev = flat_depth_bev.permute((0, 2, 1))
                flat_depth_bev = torch.matmul( flat_depth_bev, self.dparam_point1[i] )
                if self.use_chain_relu:
                    flat_depth_bev = torch.relu(flat_depth_bev)
                flat_depth_bev = torch.matmul( flat_depth_bev, self.dparam_point11[i] )
                if self.training or B > 1:
                    flat_depth_bev = flat_depth_bev.permute((0, 1, 3, 2))
                else:
                    flat_depth_bev = flat_depth_bev.permute((0, 2, 1))

                # flat_bev_col.append(flat_bev + flat_depth_bev)
                bev_img_fp = self.bev_quant(flat_bev)
                bev_depth_fp = self.depth_bev_quant(flat_depth_bev)
                fused = bev_img_fp * bev_depth_fp
                if self.use_bev_bias:
                    fused = fused + self.bev_bias[i]
                flat_bev_col.append(fused)
        elif self.learning_point_version == '1':
            for i in range(self.num_views):
                img_feat = feat[i]
                depth_feat = dfeat[i]

                flat_bev = torch.matmul( img_feat, self.param_point2[i] )
                if self.use_chain_relu:
                    flat_bev = torch.relu(flat_bev)
                if self.training or B > 1:
                    flat_bev = flat_bev.permute((0, 1, 3, 2))
                else:
                    flat_bev = flat_bev.permute((0, 2, 1))
                flat_bev = torch.matmul( flat_bev, self.param_point1[i] )
                if self.training or B > 1:
                    flat_bev = flat_bev.permute((0, 1, 3, 2))
                else:
                    flat_bev = flat_bev.permute((0, 2, 1))
                    

                flat_depth_bev = torch.matmul( depth_feat, self.dparam_point2[i] )
                if self.use_chain_relu:
                    flat_depth_bev = torch.relu(flat_depth_bev)
                if self.training or B > 1:
                    flat_depth_bev = flat_depth_bev.permute((0, 1, 3, 2))
                else:
                    flat_depth_bev = flat_depth_bev.permute((0, 2, 1))
                flat_depth_bev = torch.matmul( flat_depth_bev, self.dparam_point1[i] )
                if self.training or B > 1:
                    flat_depth_bev = flat_depth_bev.permute((0, 1, 3, 2))
                else:
                    flat_depth_bev = flat_depth_bev.permute((0, 2, 1))

                # flat_bev_col.append(flat_bev + flat_depth_bev)
                bev_img_fp = self.bev_quant(flat_bev)
                bev_depth_fp = self.depth_bev_quant(flat_depth_bev)
                fused = bev_img_fp * bev_depth_fp
                if self.use_bev_bias:
                    fused = fused + self.bev_bias[i]
                flat_bev_col.append(fused)
        elif self.learning_point_version == '9':                # 333SplitMatrix_SameWeight
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

                collect__ = self.bnMergeFD_col[i](torch.concat([flat_bev_ret0, flat_bev_ret1, flat_bev_ret2], dim = 1))
                if self.use_bev_bias:
                    collect__ = collect__ + self.bev_bias[i]
                flat_bev_col.append(collect__)
            return self.bnMergeALL(torch.concat(flat_bev_col, dim = 1))
        elif self.learning_point_version == '10':                # 333SplitMatrix_SameWeight 3muchmuchbetter
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
                if self.use_bev_bias:
                    fused = fused + self.bev_bias[i]
                flat_bev_col.append(fused)# / (self.num_views))
            trans_feat = self.bnMergeALL(torch.concat(flat_bev_col, dim = 1))
            return trans_feat
        elif self.learning_point_version == '11':                # 333SplitMatrix_SameWeight 3muchbetter
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
                flat_bev = torch.matmul(self.param_point_222[i], flat_bev)
                flat_bev = flat_bev.permute((0, 1, 3, 2))

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
                if self.use_bev_bias:
                    fused = fused + self.bev_bias[i]
                flat_bev_col.append(fused)# / (self.num_views))
            trans_feat = self.bnMergeALL(torch.concat(flat_bev_col, dim = 1))
            return trans_feat
        trans_feat = flat_bev_col[0]
        for f in flat_bev_col[1:]:
            trans_feat = self.floatFs.add(trans_feat, f)
        if self.training or B > 1:
            return trans_feat
        else:
            trans_feat = trans_feat.unsqueeze(0)
            return trans_feat

    def _spatial_transfom(self, feats: Tensor, points: Tensor) -> Tensor:
        """Apply spatial transformation to the features using the given points.

        Args:
            feats: Tuple of feature tensor and depth tensor.
            points: Tuple of feature points and depth points.

        Returns:
            The transformed feature tensor.
        """
        feat, dfeat = feats # torch.Size([8, 64, 50, 50])      torch.Size([8, 64, 50, 50])
        fpoints, dpoints = points
        (U_rank, Vh) = fpoints
        (U_rank_d, Vh_d) = dpoints

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

        homo_feats = []
        for j in range(len(U_rank)):
            for i in range(self.num_views):
                # 步骤1：展平图像特征 [B, C, H*W]

                img_feat = feat[i].view(B, -1, H * W)
                depth_feat = dfeat[i].view(B, -1, H * W)

                # 步骤2：矩阵乘法完成视图变换（核心！替代gather+scatter）
                # W_proj: [N_bev, N_pixel] → 转置后 [N_pixel, N_bev]

                flat_bev = torch.matmul(img_feat, Vh[j].T)
                flat_bev = torch.matmul(flat_bev, U_rank[j].T)
                flat_depth_bev = torch.matmul(depth_feat, Vh_d[j].T)
                flat_depth_bev = torch.matmul(flat_depth_bev, U_rank_d[j].T)

                # flat_bev = self.tile_matmul(img_feat, U_rank[j], Vh[j])
                # flat_depth_bev = self.tile_matmul(depth_feat, U_rank_d[j], Vh_d[j])
            
                # 步骤3：恢复BEV空间维度 [B, C, BEV_H, BEV_W]
                flat_bev = flat_bev.reshape(B, -1, self.grid_size[0], self.grid_size[1])
                flat_depth_bev = flat_depth_bev.reshape(B, -1, self.grid_size[0], self.grid_size[1])
                
                homo_feats.append(flat_bev * flat_depth_bev)
        
        trans_feat = homo_feats[0]
        for f in homo_feats[1:]:
            trans_feat = self.floatFs.add(trans_feat, f)
        return trans_feat

    def tile_matmul(self, img_feat, U_rank, Vh_rank):
        feat_Vh = torch.matmul(img_feat, Vh_rank.T)

        if self.num_tile == 1:
            return torch.matmul(feat_Vh, U_rank.T)

        output = []
        for i in range(self.num_tile):
            start = i * self.tile_size 
            end = (i + 1) * self.tile_size

            tile_U_rank = U_rank[start:end, :]
            flat_bev = torch.matmul(feat_Vh, tile_U_rank.T)
            output.append(flat_bev)
        out = torch.concat(output, dim = 2)
        return out

    def _spatial_transfom_origin(self, feats: Tensor, points: Tensor) -> Tensor:
        """Apply spatial transformation to the features using the given points.

        Args:
            feats: Tuple of feature tensor and depth tensor.
            points: Tuple of feature points and depth points.

        Returns:
            The transformed feature tensor.
        """
        feat, dfeat = feats
        fpoints, dpoints = points
        fpoints = self.quant_stub(fpoints)
        dpoints = self.dquant_stub(dpoints)

        B = feat.shape[0] // self.num_views
        C, H, W = feat.shape[1:]

        if self.training or B > 1:
            feat = feat.view(B, self.num_views, C, H, W)
            feat = feat.permute(0, 2, 1, 3, 4).contiguous()
        else:
            feat = feat.permute(1, 0, 2, 3).contiguous()

        feat = feat.view(B, C, -1, W) # self.num_views * H, W torch.Size([1, 64, 200, 50])  torch.Size([10, 128, 128, 2])

        dfeat = dfeat.view(B, 1, -1, H * W) # self.depth * self.num_views, H * W   torch.Size([1, 1, 256, 2500])  torch.Size([10, 128, 128, 2])
        homo_feats = []
        for i in range(self.num_points):
            homo_feat = self.grid_sample(
                feat,
                fpoints[i * B : (i + 1) * B],
            )

            homo_dfeat = self.dgrid_sample(
                dfeat,
                dpoints[i * B : (i + 1) * B],
            )
            homo_feat = self.floatFs.mul(homo_feat, homo_dfeat)
            homo_feats.append(homo_feat)

        trans_feat = homo_feats[0]
        for f in homo_feats[1:]:
            trans_feat = self.floatFs.add(trans_feat, f)
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


class GKTMultiHeadAttentionFisheye(nn.Module):
    """The GKT multi head attention.

    Args:
        embed_dims: Dims for transformer.
        nhead: number of head.
        dropout: dropout rate.
    """

    def __init__(self, embed_dims: int, nhead: int = 8, dropout: float = 0.0):
        super().__init__()
        self.q = ConvModule2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )

        self.k = ConvModule2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )

        self.v = ConvModule2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )

        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(0)

        self.q_k_mul = FloatFunctional()
        self.q_k_sum = FloatFunctional()
        self.att_v_mul = FloatFunctional()
        self.v_sum = FloatFunctional()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Forward pass of a self-attention mechanism.

        Args:
            q: Query vectors.
            k: Key vectors.
            v: Value vectors.

        Returns:
            x: The final output tensor after the self-attention operation.
        """
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        attention = self.q_k_mul.mul(q, k)
        attention = self.q_k_sum.sum(attention, dim=1, keepdim=True)
        attention = self.softmax(attention)

        x = self.att_v_mul.mul(attention, v)
        x = self.v_sum.sum(x, dim=0, keepdim=True)

        return x


class GKTTransformerLayerFisheye(nn.Module):
    """The GKT transformer layer.

    Args:
        embed_dims: Dims for transformer.
        dropout: dropout rate.
    """

    def __init__(self, embed_dims: int, dropout: float = 0.1):
        super().__init__()

        self.multihead_attn = GKTMultiHeadAttentionFisheye(embed_dims)

        self.linear1 = ConvModule2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.linear2 = ConvModule2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = hnn.LayerNorm(normalized_shape=(embed_dims, 1, 1), dim=1)
        self.norm2 = hnn.LayerNorm(normalized_shape=(embed_dims, 1, 1), dim=1)
        self.add1 = FloatFunctional()
        self.add2 = FloatFunctional()
        self.act = nn.ReLU(inplace=True)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Forward pass through the transformer block.

        Args:
            q: The query tensor.
            k: The key tensor.
            v: The value tensor.

        Returns:
            tgt: The output tensor after passing through the transformer block.
        """
        norm_q = self.norm1(q)
        tgt = self.multihead_attn(norm_q, k, v)

        tgt = self.add1.add(self.dropout1(tgt), norm_q)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout2(self.act(self.linear1(tgt))))

        tgt = self.add2.add(tgt, self.dropout3(tgt2))
        return tgt


@OBJECT_REGISTRY.register
class GKTTransformerFisheye(ViewTransformerFisheye):
    """The GKT view transform for converting image view to bev view.

    Args:
        kernel_size: Kernel size for points.
        embed_dims: Dims for transformer.
    """

    def __init__(
        self,
        kernel_size: Tuple[float] = (3, 3),
        embed_dims: int = 160,
        grid_size: Tuple[float] = None,
        **kwargs,
    ):
        super(GKTTransformerFisheye, self).__init__(grid_size=grid_size, **kwargs)
        self.kernel_size = kernel_size
        if grid_size is None:
            grid_size = (64, 64)

        self.gkt_layer = GKTTransformerLayerFisheye(embed_dims)
        query_pos_embed = torch.zeros((1, embed_dims, *grid_size))
        self.query_pos_embed = nn.Parameter(
            query_pos_embed, requires_grad=True
        )
        self.floatFs = FloatFunctional()
        self.floatFs2 = FloatFunctional()
        self.query_quant_stub = QuantStub()

    def _get_points_from_meta(self, meta: Dict) -> List[Tensor]:
        """Get points from the metadata dictionary and convert them to tensors.

        Args:
            meta: The metadata dictionary containing the points as values.

        Returns:
            points: A list of tensors representing the points.
        """
        points = []
        for k in meta.keys():
            if k.startswith("points"):
                points.append(self._convert_p2tensor(meta[k]))
        return points

    def _gen_coords_from_kernel(self, coords: Tensor) -> Tensor:
        """Generate new coordinates.

        Args:
            coords: The input coordinates.

        Returns:
            kernel_coords: The new tensor of coordinates
                           with kernel offsets applied.
        """
        h = self.kernel_size[0] - 2
        w = self.kernel_size[1] - 2
        kernel_coords = []
        for i in range(-h, h + 1):
            for j in range(-w, w + 1):
                new_coords = coords.clone()
                new_coords[..., 0] += j
                new_coords[..., 1] += i
                kernel_coords.append(new_coords)
        kernel_coords = torch.stack(kernel_coords)
        return kernel_coords

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
        new_coords = []
        for homo in homography:
            new_coord = torch.matmul(coords, homo.permute((1, 0))).float()
            new_coords.append(new_coord)
        new_coords = torch.stack(new_coords, dim=0)
        new_coords[..., 2] = torch.clamp(new_coords[..., 2], min=0.05)
        X = new_coords[..., 0] / new_coords[..., 2]
        Y = new_coords[..., 1] / new_coords[..., 2]
        new_coords = torch.stack((X, Y), dim=-1)
        new_coords = self._gen_coords_from_kernel(new_coords)
        new_coords = adjust_coords(new_coords, self.grid_size)
        new_coords = torch.unbind(new_coords, dim=0)
        return new_coords

    def _spatial_transfom(self, feats: Tensor, points: Tensor) -> Tensor:
        """Apply spatial transformation to the input features.

        Using the provided reference points and return the fused features
        after applying a Graph Kernel Transformer (GKT) layer.

        Args:
            feats: The input features.
            points: The reference points.

        Returns:
            fused_feats: The fused features after spatial
                         transformation and GKT layer.
        """
        num_points = self.kernel_size[0] * self.kernel_size[1]
        N, C, _, _ = feats.shape
        H, W = self.grid_size
        B = N // self.num_views
        trans_feats = []
        for i in range(num_points):
            trans_feat = self.grid_sample(
                feats,
                self.quant_stub(points[i]),
            )
            if B > 1:
                trans_feat = trans_feat.view(B, self.num_views, C, H, W)
                trans_feat = self.floatFs.sum(
                    trans_feat, dim=1, keepdim=True
                ).squeeze()
            else:
                trans_feat = self.floatFs.sum(trans_feat, dim=0, keepdim=True)
            trans_feats.append(trans_feat)
        trans_feats = self.floatFs.cat(trans_feats)

        query_pos_embed = self.query_quant_stub(self.query_pos_embed)
        if B > 1:
            fused_feats = []
            trans_feats = trans_feats.view(num_points, B, C, H, W).permute(
                1, 0, 2, 3, 4
            )
            for i in range(B):
                fused_feat = self.gkt_layer(
                    query_pos_embed, trans_feats[i], trans_feats[i]
                )
                fused_feats.append(fused_feat)
            fused_feats = self.floatFs2.cat(fused_feats)
        else:
            fused_feats = self.gkt_layer(
                query_pos_embed, trans_feats, trans_feats
            )
        return fused_feats

    def fuse_model(self) -> None:
        pass
#（注：内容由AI生成）
