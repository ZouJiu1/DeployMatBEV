# Copyright (c) Horizon Robotics. All rights reserved.

import collections
from typing import Any, List

import numpy as np
import torch

from hat.data.collates.collates import default_collate

__all__ = ["collate_nuscenes", "collate_nuscenes_sequence"]


def collate_nuscenesFisheye(batch: List[Any]):
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Args:
        batch: list of data.
    """
    elem = batch[0]
    # these key-value will skip default_collate
    cam_keys = ["img", "ego2img", "lidar2img", "cam2ego", "cam2lidar", \
                "lidar2cam", "lidar2img", "camera_intrinsic"]
    list_keys = [
        "img_name",
        "cam_name",
        "bev_bboxes",
        "bev_cat_ids",
        "bev_bboxes_labels",
        "layout",
        "color_space",
        "meta",
        "sample_token",
        "ego2global",
        "ego_bboxes_labels",
        "lidar_bboxes_labels",
        "center2ds",
        "corner2ds",
        "mono_3d_bboxes",
        "mono_3d_labels",
        "depths",
        "attr_labels",
        "instance_ids",
        "gt_labels_map",
        "gt_instances",
        "gt_pv_seg_mask",
        "osm_vectors",
        "gt_boxes_3d",
        'gt_label',
    ]
    if not isinstance(elem, dict):
        return default_collate(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return_data = {}
        for key in elem:
            if key in list_keys:
                collate_data = [d[key] for d in batch]
            elif key in cam_keys:
                collate_data = []
                for d in batch:
                    for item in d[key]:
                        if isinstance(item, np.ndarray):
                            item = torch.tensor(item)
                        collate_data.append(item)

                collate_data = torch.stack(collate_data, dim=0)
            elif key == "points":
                collate_data = []
                for d in batch:
                    points = torch.tensor(d[key])
                    collate_data.append(points)
            else:
                collate_data = default_collate([d[key] for d in batch])
            return_data.update({key: collate_data})

        return return_data


def collate_nuscenes_sequence(batch: List[Any]):
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Args:
        batch: list of data.
    """
    keep_keys = ["img", "ego2img", "ego2global"]
    data = []

    elem = batch[0][0]

    for b in batch:
        data_per_batch = {}
        for key in elem:
            if key in keep_keys:
                key_data = []
                for s in b:
                    if isinstance(s[key], list):
                        key_data.extend(s[key])
                    else:
                        key_data.append(s[key])
            else:
                key_data = b[0][key]
            data_per_batch[key] = key_data
        data.append(data_per_batch)

    return collate_nuscenes(data)


def collate_nuscenes_sequencev2(batch: List[Any]):
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Args:
        batch: list of data.
    """
    cam_keys = [
        "img",
        "cam2ego",
        "camera_intrinsic",
        "can_bus",
        "gt_depth",
        "gt_depth_mask",
        "depth_map",
        "depth_map_mask",
    ]
    list_keys = [
        "img_name",
        "cam_name",
        "bev_bboxes",
        "bev_cat_ids",
        "bev_bboxes_labels",
        "layout",
        "color_space",
        "meta",
        "sample_token",
        "ego2global",
        "ego_bboxes_labels",
        "lidar_bboxes_labels",
        "lidar",
        "prev_bev_exists",
        "lidar2img",
        "can_bus",
        "img_shape",
        "ego2img",
        "cams_monos",
        "cams_monos_idx",
        "pad_shape",
        "scene",
        "img_metas",
        "gt_labels_map",
        "gt_instances",
        "gt_seg_mask",
        "gt_pv_seg_mask",
        "lidar2global",
        "osm_vectors",
        "gt_occ_info",
    ]
    stack_keys = ["osm_mask"]

    elem = batch[0][0]
    seq_data = {}
    seq_data["seq_meta"] = []
    seq_length = len(batch[0])
    for _ in range(seq_length):
        seq_data["seq_meta"].append({})
    for key in elem:
        if key in list_keys:
            for idx in range(seq_length):
                if key == "can_bus":
                    collate_data = [
                        torch.tensor(d[idx][key], dtype=torch.float32)
                        for d in batch
                    ]
                    collate_data = torch.stack(collate_data, dim=0)
                elif key == "gt_occ_info":
                    collate_data = {}
                    for k2 in batch[0][idx][key]:
                        collate_data[k2] = torch.stack(
                            [
                                torch.tensor(
                                    d[idx][key][k2], dtype=torch.uint8
                                )
                                for d in batch
                            ]
                        )
                else:
                    collate_data = [d[idx][key] for d in batch]
                seq_data["seq_meta"][idx][key] = collate_data
        elif key in cam_keys:
            collate_data = []
            for idx in range(seq_length):
                for d in batch:
                    for item in d[idx][key]:
                        if isinstance(item, np.ndarray):
                            item = torch.tensor(item)
                        collate_data.append(item)
            collate_data = torch.stack(collate_data, dim=0)
            seq_data[key] = collate_data
        elif key == "points":
            collate_data = []
            for idx in range(seq_length):
                for d in batch:
                    points = torch.tensor(d[idx][key])
                    collate_data.append(points)
            seq_data[key] = collate_data
        elif key in stack_keys:
            if elem[key] is not None:
                collate_data = []
                for d in batch:
                    item = d[0][key]
                    collate_data.append(item)
                collate_data = torch.stack(collate_data, dim=0)
                seq_data[key] = collate_data
            else:
                seq_data[key] = None

    return seq_data
