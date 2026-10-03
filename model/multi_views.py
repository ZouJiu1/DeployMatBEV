# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Mapping, Optional, Sequence, Tuple

import cv2
import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import nn

from hat.core.box3d_utils import points_cam2img, points_img2cam
from hat.core.cam_box3d import CameraInstance3DBoxes
from hat.core.nus_box3d_utils import get_min_max_coords
from hat.registry import OBJECT_REGISTRY
from hat.data.transforms.grid_mask import GridMask

__all__ = [
    "BevFeatureRotate_Fisheye",
]

PIL_INTERP_CODES = {
    "nearest": F.InterpolationMode.NEAREST,
    "bilinear": F.InterpolationMode.BILINEAR,
}


@OBJECT_REGISTRY.register
class BevFeatureRotate_Fisheye(object):
    """Rotate feat.

    Args:
        bev_size: Size of bev view.
        rot: Rotate radian.
    """

    def __init__(
        self,
        bev_size: Tuple[float, float, float],
        rot: Tuple[float, float] = (-0.3925, 0.3925),
    ):
        self.rot = rot
        self.grid_sample = hnn.GridSample(
            mode="bilinear", padding_mode="zeros"
        )
        self.bev_size = bev_size

    def _get_rot(self, rot):
        return torch.Tensor(
            [
                [np.cos(rot), np.sin(rot)],
                [-np.sin(rot), np.cos(rot)],
            ]
        )

    def _get_coords(self, rot, feat):
        H, W = feat.shape[2:]
        view = np.eye(3)
        A = self._get_rot(-rot)
        B = torch.Tensor((W - 1, H - 1)) / 2
        view[:2, :2] = A
        view[:2, 2] = A @ -B + B
        view = torch.Tensor(view)

        x = (torch.linspace(0, W - 1, W).reshape((1, W)).repeat(H, 1)).float()
        y = (torch.linspace(0, H - 1, H).reshape((H, 1)).repeat(1, W)).float()
        ones = torch.ones((H, W)).float()
        coords = torch.stack([x, y, ones], dim=-1)
        coords = coords.view(1, H, W, 3)
        new_coords = torch.matmul(coords, view.T)[..., :2]
        new_coords -= coords[..., :2]
        return new_coords, view

    def _rotate_bbox(self, bbox, rot):
        min_x, max_x, min_y, max_y = get_min_max_coords(self.bev_size)
        H = max_x * 2 / self.bev_size[2]
        W = max_y * 2 / self.bev_size[2]
        view = np.eye(3)
        A = self._get_rot(rot)
        B = torch.Tensor((W, H)) / 2
        view[:2, :2] = A
        view[:2, 2] = A @ -B + B
        view = torch.Tensor(view)

        bbox = torch.Tensor(bbox)
        center = torch.cat([bbox[:2], torch.ones([1])])
        center = torch.matmul(center, view.T)
        bbox[:2] = center[:2]
        rot = bbox[6] + rot
        bbox[6] = rot
        vel = torch.cat([bbox[7:9], torch.zeros([1])])
        vel = torch.matmul(vel, view.T)
        bbox[7:9] = vel[:2]
        return bbox

    def __call__(self, feats, data: Mapping):
        batch_size = feats.shape[0]
        coords = []
        with torch.no_grad():
            for b in range(batch_size):
                rot = np.random.uniform(*self.rot)
                new_coords, view = self._get_coords(rot, feats)
                new_coords = new_coords.to(device=feats.device)
                coords.append(new_coords)
                if "bev_seg_indices" in data:
                    bev_seg_indices = data["bev_seg_indices"][b : b + 1]
                    bev_seg_indices = bev_seg_indices.unsqueeze(1)
                    new_coords, _ = self._get_coords(rot, bev_seg_indices)
                    new_coords = new_coords.to(device=feats.device)
                    bev_seg_indices = self.grid_sample(
                        bev_seg_indices.float(), new_coords
                    ).int()
                    bev_seg_indices = bev_seg_indices.squeeze()
                    data["bev_seg_indices"][b] = bev_seg_indices
                if "bev_bboxes_labels" in data:
                    for i, bbox in enumerate(data["bev_bboxes_labels"][b]):
                        data["bev_bboxes_labels"][b][i] = self._rotate_bbox(
                            bbox, rot
                        )

        coords = torch.cat(coords)
        feats = self.grid_sample(feats, coords)

        return feats, data

    def __repr__(self):
        return "NuscBevRotate"
