# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List, Sequence

import numpy as np
import torch
from torch import nn

from hat.core.center_utils import (
    draw_umich_gaussian,
    draw_umich_gaussian_torch,
    gaussian_radius,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import multi_apply

__all__ = ["CenterPointTarget"]


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


@OBJECT_REGISTRY.register
class CenterPointTarget_Fisheye(nn.Module):
    """Generate centerpoint targets for bev task.

    Args:
        class_names: List of class names for bev detection.
        tasks: List of tasks
        gaussian_overlap: Gaussian overlap for genenrate heatmap target.
        min_radius: Min values for radius.
        out_size_factor: Output size for factor.
        norm_bbox: Whether using norm bbox.
        max_num: Max number for bbox.
        bbox_weight: Weight for bbox meta.
    """

    def __init__(
        self,
        class_names: Sequence[str],
        tasks: Sequence[dict],
        gaussian_overlap: float = 0.1,
        min_radius: int = 2,
        out_size_factor: int = 4,
        norm_bbox: bool = True,
        max_num: int = 500,
        bbox_weight: float = None,
        use_heatmap: bool = True,
    ):
        super(CenterPointTarget_Fisheye, self).__init__()

        self.class_names = class_names
        self.out_size_factor = out_size_factor
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.norm_bbox = norm_bbox
        self.max_num = max_num
        self.tasks = tasks
        self.use_heatmap = use_heatmap

        if bbox_weight is None:
            self.bbox_weight = bbox_weight
        else:
            self.bbox_weight = [1.0 for _ in range(10)] # no vel
            self.bbox_weight[-1] = 0.2
            self.bbox_weight[-2] = 0.2

    def _get_task_targets(self, gt_bboxes_3d, preds, task):
        heatmap_pred = preds["heatmap"]
        reg_pred = preds["reg"]
        height_pred = preds["height"]
        dim_pred = preds["dim"]
        rot_pred = preds["rot"]
        # preds["vel"] = torch.zeros_like(rot_pred, requires_grad=False).detach()
        if "vel" in preds:
            vel_pred = preds["vel"]
        else:
            vel_pred = None
        heatmaps, indices, bbox_targets_list = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            heatmap_pred,
            reg_pred,
            dim_pred,
            rot_pred,
            vel_pred,
            task=task,
            max_obj=self.max_num,
        )

        heatmaps = torch.stack(heatmaps)
        heatmaps_target = {
            "logits": clip_sigmoid(heatmap_pred),
            "labels": heatmaps,
        }

        if vel_pred is None:
            bbox_pred = torch.cat(
                [reg_pred, height_pred, dim_pred, rot_pred], dim=1
            )
        else:
            bbox_pred = torch.cat(
                [reg_pred, height_pred, dim_pred, rot_pred, vel_pred], dim=1
            )

        bbox_pred = torch.permute(bbox_pred, (0, 2, 3, 1)).contiguous()
        if self.use_heatmap is True:
            pos_bbox_targets = torch.stack(bbox_targets_list)
            if self.bbox_weight:
                bbox_weight = torch.tensor(self.bbox_weight).to(
                    device=heatmap_pred.device
                )
            else:
                bbox_weight = torch.ones(bbox_pred.shape[3]).to(
                    device=heatmap_pred.device
                )
            bbox_weight_heatmap = torch.zeros_like(pos_bbox_targets[..., 0])
            num_cls = heatmaps.shape[1]
            for i in range(num_cls):
                heatmaps_weight = heatmaps[:, i]
                torch.max(
                    heatmaps_weight,
                    bbox_weight_heatmap,
                    out=bbox_weight_heatmap,
                )

            avg_factor = max(bbox_weight_heatmap.sum(), 1)

            bbox_weight_heatmap = (
                bbox_weight_heatmap.unsqueeze(-1) * bbox_weight
            )
            bbox_weight = bbox_weight_heatmap
            pos_bbox_pred = bbox_pred
            refact_indices = []
        else:
            refact_indices = []
            pos_bbox_targets = []
            for i in range(len(indices)):
                if len(indices[i]) != 0:
                    pos_bbox_targets.append(bbox_targets_list[i])
                for index in indices[i]:
                    ref_index = torch.tensor([i, *index])
                    refact_indices.append(ref_index)

            if len(refact_indices) == 0:
                bbox_weight = torch.zeros(bbox_pred.shape[3]).to(
                    device=bbox_pred.device
                )
                refact_indices = torch.zeros((1, 3)).long()
                pos_bbox_targets = torch.zeros((1, bbox_pred.shape[3])).to(
                    device=bbox_pred.device
                )
            else:
                refact_indices = torch.stack(refact_indices)
                pos_bbox_targets = torch.cat(pos_bbox_targets)
                if self.bbox_weight:
                    bbox_weight = torch.tensor(self.bbox_weight).to(
                        device=heatmap_pred.device
                    )
                else:
                    bbox_weight = torch.ones(bbox_pred.shape[3]).to(
                        device=heatmap_pred.device
                    )
            pos_bbox_pred = bbox_pred[
                refact_indices[:, 0],
                refact_indices[:, 1],
                refact_indices[:, 2],
            ]
            avg_factor = pos_bbox_pred.shape[0]

        bbox_targets = {
            "pred": pos_bbox_pred,
            "target": pos_bbox_targets,
            "weight": bbox_weight,
            "avg_factor": avg_factor,
        }
        return {
            "task_name": task["name"],
            "cls_target": heatmaps_target,
            "reg_target": bbox_targets,
            "pos_indices": refact_indices,
        }

    def _gen_offset_map(self, bbox_target, center, radius):
        center = center  # .cpu().numpy()
        center_int = center.int()  # (int(center[0]), int(center[1]))
        center_offset = (
            center - center_int
        )  # np.array(center) - np.array(center_int)
        x, y = center_int

        y_grid = torch.arange(y - radius, y + radius + 1).to(
            device=center.device
        )
        x_grid = torch.arange(x - radius, x + radius + 1).to(
            device=center.device
        )
        y_reg = center[1] - y_grid
        x_reg = center[0] - x_grid

        y_reg[radius] = center_offset[1]
        x_reg[radius] = center_offset[0]
        xv, yv = torch.meshgrid(x_reg, y_reg)
        ct_off_reg_map = torch.stack([xv, yv], dim=-1).to(device=center.device)

        self._gen_heatmap(ct_off_reg_map, bbox_target, center, radius, 0, 2)

    def _gen_heatmap(self, src_map, heatmap, center, radius, start, end):
        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        rh, rw = src_map.shape[:2]
        ry, rx = (rh - 1) // 2, (rw - 1) // 2

        heatmap[
            y - top : y + bottom, x - left : x + right, start:end
        ] = src_map[ry - top : ry + bottom, rx - left : rx + right]

    def _gen_reg_map(self, value, bbox_target, center, radius, start, end):
        h = w = radius * 2 + 1
        src_map = torch.tile(value, (h, w, 1))
        self._gen_heatmap(src_map, bbox_target, center, radius, start, end)

    def get_targets_single(
        self,
        gt_bboxes_3d,
        heatmap_pred,
        reg_pred,
        dim_pred,
        rot_pred,
        vel_pred,
        task,
        max_obj,
    ):
        feat_size = heatmap_pred.shape[1:]
        # reorganize the gt_dict by tasks
        gt_bboxes_task = []
        if len(gt_bboxes_3d) != 0:
            for cls in task["class_names"]:
                cat_id = self.class_names.index(cls)
                task_indices = gt_bboxes_3d[:, -1] == cat_id
                gt_bbox = gt_bboxes_3d[task_indices]
                gt_bboxes_task.append(gt_bbox)

        indices = []
        if self.use_heatmap is True:
            bbox_dim = 10 if vel_pred is not None else 9
            bbox_targets = torch.zeros(
                (feat_size[0], feat_size[1], bbox_dim)
            ).to(device=heatmap_pred.device)
        else:
            bbox_targets = []
        heatmaps = []

        for _, gt_bbox in enumerate(gt_bboxes_task):
            heatmap = np.zeros((feat_size[0], feat_size[1]))
            for bbox in gt_bbox:
                width, length = bbox[3:5] / self.out_size_factor
                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width), min_overlap=self.gaussian_overlap
                    )
                    radius = max(self.min_radius, int(radius))
                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y = bbox[:2] / self.out_size_factor
                    z = bbox[2]
                    hi = torch.tensor([z]).to(device=heatmap_pred.device)
                    center = torch.tensor(
                        [x, y], dtype=torch.float32, device=heatmap_pred.device
                    )
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (
                        0 <= center_int[0] < feat_size[1]
                        and 0 <= center_int[1] < feat_size[0]
                    ):
                        continue
                    heatmap = draw_umich_gaussian(
                        heatmap, center_int.cpu().numpy(), radius
                    )

                    x, y = center_int[0], center_int[1]

                    assert y * feat_size[1] + x < feat_size[0] * feat_size[1]
                    indices.append([y, x])
                    reg = center - center_int
                    box_dim = torch.tensor(bbox[3:6]).to(
                        device=heatmap_pred.device
                    )
                    if self.norm_bbox:
                        box_dim = torch.log(box_dim)

                    rot = torch.tensor(bbox[6])
                    rot_sine = (
                        torch.sin(rot).view(-1).to(device=heatmap_pred.device)
                    )
                    rot_cos = (
                        torch.cos(rot).view(-1).to(device=heatmap_pred.device)
                    )

                    if self.use_heatmap is True:
                        self._gen_offset_map(bbox_targets, center, radius)
                        self._gen_reg_map(
                            hi, bbox_targets, center, radius, 2, 3
                        )
                        self._gen_reg_map(
                            box_dim, bbox_targets, center, radius, 3, 6
                        )
                        self._gen_reg_map(
                            rot_sine, bbox_targets, center, radius, 6, 7
                        )
                        self._gen_reg_map(
                            rot_cos, bbox_targets, center, radius, 7, 8
                        )
                        if vel_pred is not None:
                            vel = torch.tensor(bbox[7:9]).to(
                                device=heatmap_pred.device
                            )
                            self._gen_reg_map(
                                vel, bbox_targets, center, radius, 8, 10
                            )
                    else:
                        if vel_pred is None:
                            bbox_target = torch.stack(
                                [reg, hi, box_dim, rot_sine, rot_cos]
                            )
                        else:
                            vel = torch.tensor(bbox[7:9]).to(
                                device=heatmap_pred.device
                            )
                            if torch.isnan(vel).any():
                                vel = torch.zeros(2).to(
                                    device=heatmap_pred.device
                                )
                            bbox_target = torch.cat(
                                [reg, hi, box_dim, rot_sine, rot_cos, vel]
                            )
                            bbox_targets.append(bbox_target)
                            if len(bbox_targets) >= self.max_num:
                                break

            heatmaps.append(heatmap)
            if self.use_heatmap is False:
                if len(bbox_targets) >= self.max_num:
                    break
        if len(heatmaps) == 0:
            heatmaps = torch.zeros(
                (len(task["class_names"]), feat_size[0], feat_size[1])
            ).to(device=heatmap_pred.device)
        else:
            heatmaps = np.stack(heatmaps)
            heatmaps = torch.tensor(heatmaps).to(device=heatmap_pred.device)

        if self.use_heatmap is False:
            if len(indices) == 0:
                bbox_targets = torch.zeros((1,))
            else:
                bbox_targets = torch.stack(bbox_targets)
        return heatmaps, indices, bbox_targets

    def forward(self, label, preds, *args):
        task_targets = []
        for task_preds, task in zip(preds, self.tasks):
            task_targets.append(
                self._get_task_targets(label, task_preds, task)
            )
        return task_targets

