# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import logging
import math
import os
import os.path as osp
import pickle
from typing import Callable, Dict, List, Optional, Tuple
import json
import cv2
import msgpack
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from hat.core import box_np_ops
from hat.core.cam_box3d import CameraInstance3DBoxes
from hat.core.nus_box3d_utils import bbox_ego2bev, get_min_max_coords, bbox_bev2ego
from hat.data.utils import decode_img
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import PackTypeMapper
from hat.utils.pack_type.utils import get_packtype_from_path
from hat.utils.package_helper import require_packages
from hat.data.datasets.data_packer import Packer
import mmengine

from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion
try:
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
    from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from shapely import affinity, ops
    from shapely.geometry import LineString, MultiLineString, MultiPolygon, box

    logger = logging.getLogger("shapely")
    logger.setLevel(logging.ERROR)
except ImportError:
    NuScenesCanBus = None
    Quaternion = None
    quaternion_yaw = None
    NuScenesMap = None
    NuScenesMapExplorer = None
    NuScenes = None
    splits = None

from lmdbdata.classesLabel import CLASSES

logger = logging.getLogger(__name__)

__all__ = [
    "fisheyeGOYUPacker",
    "FisheyeGOYUFromImage",
    "FisheyeGOYUFromImageSequence",
    "FisheyeGOYUMonoDataset",
    "FisheyeGOYUMonoFromImage",
    "create_nuscenes_groundtruth_database",
    "FisheyeBevDataset",
]

# NameMapping = {
#     "movable_object.barrier": "barrier",
#     "vehicle.bicycle": "bicycle",
#     "vehicle.bus.bendy": "bus",
#     "vehicle.bus.rigid": "bus",
#     "vehicle.car": "car",
#     "vehicle.construction": "construction_vehicle",
#     "vehicle.motorcycle": "motorcycle",
#     "human.pedestrian.adult": "pedestrian",
#     "human.pedestrian.child": "pedestrian",
#     "human.pedestrian.construction_worker": "pedestrian",
#     "human.pedestrian.police_officer": "pedestrian",
#     "movable_object.trafficcone": "traffic_cone",
#     "vehicle.trailer": "trailer",
#     "vehicle.truck": "truck",
# }

NameMapping = {}
for i in CLASSES:
    NameMapping[i] = i

DefaultAttribute = {
    "car": "vehicle.parked",
    "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked",
    "truck": "vehicle.parked",
    "bus": "vehicle.moving",
    "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "barrier": "",
    "traffic_cone": "",
}

Attributes = (
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
    "None",
)


NUSCENES_SEMANTIC_MAPPING = {
    1: 0,
    5: 0,
    7: 0,
    8: 0,
    10: 0,
    11: 0,
    13: 0,
    19: 0,
    20: 0,
    0: 0,
    29: 0,
    31: 0,
    9: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    21: 0,
    2: 0,
    3: 0,
    4: 0,
    6: 0,
    12: 0,
    22: 0,
    23: 0,
    24: 1,
    25: 0,
    26: 0,
    27: 0,
    28: 0,
    30: 0,
}


def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(
        patch, patch_angle, origin=(patch_x, patch_y), use_radians=False
    )

    return patch


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def mask_for_lines(lines, mask, thickness, idx, type="index", angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)
    coords = coords.reshape((-1, 2))
    if len(coords) < 2:
        return mask, idx
    if type == "backward":
        coords = np.flip(coords, 0)

    if type == "index":
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else:
        for i in range(len(coords) - 1):
            cv2.polylines(
                mask,
                [coords[i:]],
                False,
                color=get_discrete_degree(
                    coords[i + 1] - coords[i], angle_class=angle_class
                ),
                thickness=thickness,
            )
    return mask, idx


def line_geom_to_mask(
    layer_geom,
    confidence_levels,
    local_box,
    canvas_size,
    thickness,
    idx,
    type="index",
    angle_class=36,
):
    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box)

    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)

    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            new_line = affinity.affine_transform(
                new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y]
            )
            new_line = affinity.scale(
                new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0)
            )
            confidence_levels.append(confidence)
            if new_line.geom_type == "MultiLineString":
                for new_single_line in new_line:
                    map_mask, idx = mask_for_lines(
                        new_single_line,
                        map_mask,
                        thickness,
                        idx,
                        type,
                        angle_class,
                    )
            else:
                map_mask, idx = mask_for_lines(
                    new_line, map_mask, thickness, idx, type, angle_class
                )
    return map_mask, idx


def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C - 1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask


def preprocess_map(
    vectors, patch_size, canvas_size, max_channel, thickness, angle_class
):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(max_channel):
        vector_num_list[i] = []

    for vector in vectors:
        if vector["pts_num"] >= 2:
            vector_num_list[vector["type"]].append(
                LineString(vector["pts"][: vector["pts_num"]])
            )

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    filter_masks = []
    instance_masks = []
    forward_masks = []
    backward_masks = []
    for i in range(max_channel):
        map_mask, idx = line_geom_to_mask(
            vector_num_list[i],
            confidence_levels,
            local_box,
            canvas_size,
            thickness,
            idx,
        )
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(
            vector_num_list[i],
            confidence_levels,
            local_box,
            canvas_size,
            thickness + 4,
            1,
        )
        filter_masks.append(filter_mask)
        forward_mask, _ = line_geom_to_mask(
            vector_num_list[i],
            confidence_levels,
            local_box,
            canvas_size,
            thickness,
            1,
            type="forward",
            angle_class=angle_class,
        )
        forward_masks.append(forward_mask)
        backward_mask, _ = line_geom_to_mask(
            vector_num_list[i],
            confidence_levels,
            local_box,
            canvas_size,
            thickness,
            1,
            type="backward",
            angle_class=angle_class,
        )
        backward_masks.append(backward_mask)

    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    forward_masks = np.stack(forward_masks)
    backward_masks = np.stack(backward_masks)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    forward_masks = (
        overlap_filter(forward_masks, filter_masks).sum(0).astype("int32")
    )
    backward_masks = (
        overlap_filter(backward_masks, filter_masks).sum(0).astype("int32")
    )

    semantic_masks = instance_masks != 0

    return semantic_masks, instance_masks, forward_masks, backward_masks


class VectorizedLocalMap(object):
    def __init__(
        self,
        dataroot,
        patch_size,
        canvas_size,
        line_classes,
        ped_crossing_classes,
        contour_classes,
        sample_dist=1,
        num_samples=250,
        padding=False,
        normalize=False,
        fixed_num=-1,
        class2label=None,
    ):
        super().__init__()
        self.data_root = dataroot
        self.MAPS = [
            "boston-seaport",
            "singapore-hollandvillage",
            "singapore-onenorth",
            "singapore-queenstown",
        ]
        if class2label is None:
            class2label = {
                "road_divider": 0,
                "lane_divider": 0,
                "ped_crossing": 1,
                "contours": 2,
                "others": -1,
            }

        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.class2label = class2label
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=self.data_root, map_name=loc
            )
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.normalize = normalize
        self.fixed_num = fixed_num

    @require_packages(
        "nuscenes", raise_msg="Please `pip3 install nuscenes-devkit`"
    )
    def gen_vectorized_samples(
        self, location, ego2global_translation, ego2global_rotation
    ):
        map_pose = ego2global_translation[:2]

        rotation = Quaternion(ego2global_rotation)

        patch_box = (
            map_pose[0],
            map_pose[1],
            self.patch_size[0],
            self.patch_size[1],
        )
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        line_geom = self.get_map_geom(
            patch_box, patch_angle, self.line_classes, location
        )
        line_vector_dict = self.line_geoms_to_vectors(line_geom)

        ped_geom = self.get_map_geom(
            patch_box, patch_angle, self.ped_crossing_classes, location
        )
        # ped_vector_list = self.ped_geoms_to_vectors(ped_geom)
        ped_vector_list = self.line_geoms_to_vectors(ped_geom)["ped_crossing"]

        polygon_geom = self.get_map_geom(
            patch_box, patch_angle, self.polygon_classes, location
        )
        poly_bound_list = self.poly_geoms_to_vectors(polygon_geom)

        vectors = []
        for line_type, vects in line_vector_dict.items():
            for line, length in vects:
                vectors.append(
                    (
                        line.astype(float),
                        length,
                        self.class2label.get(line_type, -1),
                    )
                )

        for ped_line, length in ped_vector_list:
            vectors.append(
                (
                    ped_line.astype(float),
                    length,
                    self.class2label.get("ped_crossing", -1),
                )
            )

        for contour, length in poly_bound_list:
            vectors.append(
                (
                    contour.astype(float),
                    length,
                    self.class2label.get("contours", -1),
                )
            )

        # filter out -1
        filtered_vectors = []
        for pts, pts_num, type in vectors:
            if type != -1:
                filtered_vectors.append(
                    {"pts": pts, "pts_num": pts_num, "type": type}
                )

        return filtered_vectors

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.map_explorer[location]._get_layer_line(
                    patch_box, patch_angle, layer_name
                )
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(
                    patch_box, patch_angle, layer_name
                )
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(
                    patch_box, patch_angle, location
                )
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == "MultiLineString":
                    for li in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(li))
                elif line.geom_type == "LineString":
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != "MultiPolygon":
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = {}
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(
                a_type_of_lines
            )
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != "MultiPolygon":
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        def add_line(
            poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list
        ):
            points = [
                (p0, p1)
                for p0, p1 in zip(
                    poly_xy[0, idx : idx + 2], poly_xy[1, idx : idx + 2]
                )
            ]
            line = LineString(points)
            line = line.intersection(patch)
            if not line.is_empty:
                line = affinity.rotate(
                    line,
                    -patch_angle,
                    origin=(patch_x, patch_y),
                    use_radians=False,
                )
                line = affinity.affine_transform(
                    line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y]
                )
                line_list.append(line)

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []

        records = self.nusc_maps[location].ped_crossing
        for record in records:
            polygon = self.map_explorer[location].extract_polygon(
                record["polygon_token"]
            )
            poly_xy = np.array(polygon.exterior.xy)
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            add_line(
                poly_xy, x1, patch, patch_angle, patch_x, patch_y, line_list
            )
            add_line(
                poly_xy, x2, patch, patch_angle, patch_x, patch_y, line_list
            )

        return line_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array(
                [
                    list(line.interpolate(distance).coords)
                    for distance in distances
                ]
            ).reshape(-1, 2)
        else:
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array(
                [
                    list(line.interpolate(distance).coords)
                    for distance in distances
                ]
            ).reshape(-1, 2)

        if self.normalize:
            sampled_points = sampled_points / np.array(
                [self.patch_size[1], self.patch_size[0]]
            )

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate(
                    [sampled_points, padding], axis=0
                )
            else:
                sampled_points = sampled_points[: self.num_samples, :]
                num_valid = self.num_samples

            if self.normalize:
                sampled_points = sampled_points / np.array(
                    [self.patch_size[1], self.patch_size[0]]
                )
                num_valid = len(sampled_points)

        return sampled_points, num_valid

class Bbox(object):
    """Bbox object for NuScenes dataset.
    Args:
        sample: Meta data for sample.
        bbox: Meta data for BBox.
    """

    def __init__(self, sample: dict, bbox: dict):
        self.sample = sample
        box = copy.deepcopy(bbox["bbox"])
        self.center = box[:3]
        self.wlh = box[3:6]
        rot = bbox["rot"]
        self.cat = bbox["cat"]
        self.token = bbox["token"]
        self.attr_id = bbox["attr_id"]
        self.rot = Quaternion(rot)
        if len(bbox['velocity']) == 3:
            self.velocity = np.array(bbox['velocity'])
        else:
            self.velocity = np.array([*bbox["velocity"], 0.0])
        if np.isnan(self.velocity).any():
            self.velocity = [0.0, 0.0, 0.0]

    def _limit_period(self, val, offset=0.5, period=np.pi * 2):
        return val - np.floor(val / period + offset) * period

    def _translate(self, x: np.array):
        self.center += x

    def get_attr_label(self):
        if self.cat in NameMapping:
            cat_id = CLASSES.index(NameMapping[self.cat])
        else:
            cat_id = -1
        return cat_id

    def _rotate(self, rot: Quaternion):
        quaternion = Quaternion(rot)
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.rot = quaternion * self.rot
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self):
        w, l, h = self.wlh

        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.rot.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def get_box_token(self):
        return self.token

    def get_bev_bboxes(self) -> np.array:
        """Get bbox for cam coordinate."""
        # ego2global_translation = self.sample["ego2global_translation"]
        # ego2global_rotation = self.sample["ego2global_rotation"]
        # self._translate(-np.array(ego2global_translation))
        # self._rotate(Quaternion(ego2global_rotation).inverse)
        
        yaw = Quaternion(self.rot).yaw_pitch_roll[0]
        yaw = np.array([self._limit_period(yaw)])
        wlh = np.array(self.wlh)
        if self.cat in NameMapping:
            cat_id = CLASSES.index(NameMapping[self.cat])
        else:
            cat_id = -1
        velocity = self.velocity[:2]
        cat_id = np.array([cat_id])
        bboxes = np.concatenate(
            [self.center, wlh, yaw, velocity, cat_id], axis=0
        )
        # without velocity
        # bboxes = np.concatenate([self.center, wlh, yaw, cat_id], axis = 0)
        return bboxes

    def get_lidar_bboxes(self):
        """Get bbox for cam coordinate."""
        sensor2ego_translation = self.sample["sensor2ego_translation"]
        sensor2ego_rotation = self.sample["sensor2ego_rotation"]

        ego2global_translation = self.sample["ego2global_translation"]
        ego2global_rotation = self.sample["ego2global_rotation"]

        self._translate(-np.array(ego2global_translation))
        self._rotate(Quaternion(ego2global_rotation).inverse)

        self._translate(-np.array(sensor2ego_translation))
        self._rotate(Quaternion(sensor2ego_rotation).inverse)

        yaw = Quaternion(self.rot).yaw_pitch_roll[0]
        yaw = np.array([self._limit_period(yaw)])
        lwh = np.array([self.wlh[1], self.wlh[0], self.wlh[2]])
        if self.cat in NameMapping:
            cat_id = CLASSES.index(NameMapping[self.cat])
        else:
            cat_id = -1
        velocity = self.velocity[:2]
        cat_id = np.array([cat_id])
        bboxes = np.concatenate(
            [self.center, lwh, yaw, velocity, cat_id], axis=0
        )
        return bboxes

    def get_cam_bboxes(self, cam):
        ego2camera = np.array(cam['ego2camera'])
        self._translate(ego2camera[:3, 3])
        # kk = np.dot(ego2camera[:3, :3], ego2camera[:3, :3].conj().transpose())
        self._rotate(Quaternion._from_matrix(ego2camera[:3, :3], rtol=1e-04, atol=1e-04))

        yaw_pitch_roll = Quaternion(self.rot).yaw_pitch_roll
        yaw = yaw_pitch_roll[0]
        yaw = np.array([self._limit_period(yaw)])
        wlh = np.array(self.wlh)
        lhw = wlh[..., [1, 2, 0]]
        if self.cat in NameMapping:
            cat_id = CLASSES.index(NameMapping[self.cat])
        else:
            cat_id = -1
        velocity = self.velocity[0::2]
        cat_id = np.array([cat_id])

        # with velocity
        bboxes = np.concatenate([ self.center, lhw, -yaw, velocity, cat_id ], axis = 0)

        return bboxes

class NuscenesDataset(Dataset):
    """Dataset object for packed NuScenes.

    Args:
        data_path: packed dataset path.
        transforms: A function transform that takes input
            sample and its target as entry and returns a transformed version.
        pack_type: The pack type.
        pack_kwargs: Kwargs for pack type.
    """

    @require_packages(
        "nuscenes", raise_msg="Please `pip3 install nuscenes-devkit`"
    )
    def __init__(
        self,
        data_path: str,
        transforms: Optional[Callable] = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
        num_split: int = 1,
    ):
        self.root = data_path
        self.transforms = transforms

        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        try:
            self.pack_type = get_packtype_from_path(data_path)
        except NotImplementedError:
            assert pack_type is not None
            self.pack_type = PackTypeMapper(pack_type.lower())

        self.pack_file = self.pack_type(
            self.root, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()
        self.meta = self._decode_meta(self.pack_file)
        # if self.meta is None:
        #     logger.warning("Packer data donot include meta. ")  # noqa [E501]
        # if self.meta:
        #     org_len = len(self.meta["scene_info"])

        #     if num_split > 1:
        #         self._split_scenes(num_split)
        #         new_len = len(self.meta["scene_info"])
        #         print(f"Split scenes from {org_len} to {new_len}")
        #     self._gen_idx2scenes()

    def _gen_idx2scenes(self):
        idx = 0
        self.idx2scenes = {}
        for k, v in self.meta["scene_info"].items():
            for _ in range(v):
                self.idx2scenes[idx] = k
                idx += 1
        # print(self.idx2scenes)

    def get_aug(self):
        aug_cfg = {}
        if self.transforms is not None:
            for trans in self.transforms.transforms:
                if hasattr(trans, "get_aug"):
                    name, cfg = trans.get_aug()
                    aug_cfg[name] = cfg
        return aug_cfg

    def set_aug(self, scene, aug_cfg):
        self.scenes_aug[scene] = aug_cfg

    def _split_scenes(self, num_split):
        new_scene_info = {}
        for k, v in self.meta["scene_info"].items():
            sequence_length = np.array(
                list(
                    range(
                        0,
                        v,
                        math.ceil(v / num_split),
                    )
                )
                + [v]
            )
            for idx, sub_seq_len in enumerate(
                sequence_length[1:] - sequence_length[:-1]
            ):
                name = k + f"_{idx}"
                new_scene_info[name] = sub_seq_len

        self.meta["scene_info"] = new_scene_info

    def get_meta(self):
        return self.meta

    def _decode_meta(self, pack_file):
        meta = self._decode(pack_file, "__meta__")
        return meta

    def _decode(self, pack_file, sample):
        def _decode_hook(obj):
            def _decode_bytes(obj):
                if isinstance(obj, bytes):
                    obj = obj.decode("utf-8")
                return obj

            new_obj = {}
            for k, v in obj.items():
                k = _decode_bytes(k)
                if k not in [
                    "img",
                    "voxel_semantics",
                    "mask_lidar",
                    "mask_camera",
                ]:
                    v = _decode_bytes(v)
                new_obj[k] = v
            return new_obj
        # print("sample = pack_file.read(sample)")
        sample = pack_file.read(sample)
        if sample is not None:
            sample = msgpack.unpackb(
                sample, object_hook=_decode_hook, raw=True
            )
            # if 'cam' in sample.keys():
                # print(sample['cam'][0]['img_path'])
        return sample

    def __getstate__(self):
        state = self.__dict__
        state["pack_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.pack_file = self.pack_type(
            self.root, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()

@OBJECT_REGISTRY.register
class FisheyeBevDataset(NuscenesDataset):
    """Bev Dataset object for packed NuScenes.

    Args:
        with_bev_bboxes: Whether include bev bboxes.
        with_ego_bboxes: Whether include ego bboxes.
        with_lidar_bboxes: Whether include lidar bboxes.
        max_dist: Maximal distance for lidar bboxes.
        with_bev_mask: Whether include bev bboxes.
        map_path: Path to Nuscenes Map, needed if include bev mask.
        line_classes: Classes of line. ex. road divider, lane divider.
        ped_crossing_classes: Classes of ped corssing. ex. ped_crossing
        contour_classes: Classes of contour. ex. road segment, lane.
        bev_size: Size for bev using meter. ex. (51.2, 51.2, 0.2)
        bev_range: range for bev, alternative of bev_size.
            ex.(-61.2, -61.2, -2, 61.2, 61.2, 10)
        map_size: Size for seg map.
        need_lidar: Whether need lidar points. Default: False.
        num_sweeps: Number of sweeps, if lidar points is needed.
                    Default: 0
        load_dim: Dimension number of the loaded points.
                  Defaults to 5.
        use_dim: Which dimension to use.
        need_mono_data: Whether need mono data. Default: False.
        with_ego_occ: Whether include ego occ.
        with_lidar_occ: Whether include lidar occ.
    """

    def __init__(
        self,
        with_bev_bboxes: bool = True,
        with_ego_bboxes: bool = False,
        with_lidar_bboxes: bool = False,
        with_bev_mask: bool = True,
        max_dist: Optional[float] = 55.0,
        map_path: Optional[str] = None,
        line_classes: Optional[List[str]] = None,
        ped_crossing_classes: Optional[List[str]] = None,
        contour_classes: Optional[List[str]] = None,
        bev_size: Optional[Tuple] = None,
        bev_range: Optional[Tuple] = None,
        map_size: Optional[Tuple] = None,
        need_lidar: Optional[bool] = False,
        num_sweeps: Optional[int] = 0,
        load_dim: Optional[int] = 5,
        use_dim: Optional[List[int]] = None,
        need_mono_data: bool = False,
        with_ego_occ: bool = False,
        with_lidar_occ: bool = False,
        **kwargs,
    ):
        super(FisheyeBevDataset, self).__init__(**kwargs)
        self.with_lidar_occ = with_lidar_occ
        self.sampler = NuscenesBevSampler(
            with_bev_bboxes=with_bev_bboxes,
            with_ego_bboxes=with_ego_bboxes,
            with_lidar_bboxes=with_lidar_bboxes,
            with_bev_mask=with_bev_mask,
            max_dist=max_dist,
            map_path=map_path,
            line_classes=line_classes,
            ped_crossing_classes=ped_crossing_classes,
            contour_classes=contour_classes,
            bev_size=bev_size,
            bev_range=bev_range,
            map_size=map_size,
            need_lidar=need_lidar,
            num_sweeps=num_sweeps,
            load_dim=load_dim,
            use_dim=use_dim,
            need_mono_data=need_mono_data,
            with_ego_occ=with_ego_occ,
        )
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     state["pack_file"] = None
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__ = state
    #     # 只重开 LMDB，不重新创建 sampler
    #     self.pack_file = self.pack_type(
    #         self.root, writable=False, **self.kwargs
    #     )
    #     self.pack_file.open()
    #     self.samples = self.pack_file.get_keys()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        aug_cfg = None
        if isinstance(item, dict):
            idx = item["idx"]
            aug_cfg = item["aug"]
        else:
            idx = item
        if self.pack_file == None:
            super(FisheyeBevDataset, self).__init__(data_path=self.root, transforms=self.transforms, \
                        pack_type="lmdb", pack_kwargs=self.kwargs)
        # sample = self._decode(self.pack_file, self.samples[idx])
        try:
            # print(idx, idx)
            sample = self._decode(self.pack_file, self.samples[idx])
        except Exception as e:
            logger.error(f"Failed to decode sample idx={idx}, key={self.samples[idx]}: {e}")
            raise
        sample = NuscenesSample(sample)
        # print("data = self.sampler(sample)")
        data = self.sampler(sample)
        # print("data = self.sampler(sample) end")
        if aug_cfg:
            data["scenes_aug"] = aug_cfg
        if self.transforms is not None:
            # print("data = self.transforms(data)")
            data = self.transforms(data)
            # print("data = self.transforms(data) end")
        # if self.with_lidar_occ:
        #     data["gt_occ_info"] = sample._get_lidar_occ(data["lidar2global"])
        return data

class NuscenesSample(object):
    """Sample object for NuScenes dataset.

    Args:
        sample: Meta data for sample.
    """

    def __init__(self, sample: dict):
        self.sample = sample

    def __del__(self):
        del self.sample

    def get_scene(self):
        return self.sample["scene"]

    def get_timestamp(self):
        return self.sample["timestamp"] / 1e6

    def get_can_bus(self):
        return self.sample["can_bus"]

    def get_bboxes_by_bev(self, bev_size: Tuple[float, float, float]) -> List:
        """Get bboxes for bev coordinate."""
        if bev_size is None:
            raise ValueError("bev_size is needed to gen bev bboxes")
        gt_bboxes = self.sample["gt_bboxes"]
        gtbox = copy.deepcopy(gt_bboxes)

        bboxes = []
        gt = []
        for idx, bbox in enumerate(gt_bboxes):
            # if bbox["num_lidar_pts"] == 0 and bbox["num_radar_pts"] == 0:
            #     continue
            box = Bbox(self.sample, bbox)
            ego_bbox = box.get_bev_bboxes()
            min_x, max_x, min_y, max_y = get_min_max_coords(bev_size)
            valid_mask = (
                (ego_bbox[0] > min_x)
                & (ego_bbox[0] < max_x)
                & (ego_bbox[1] > min_y)
                & (ego_bbox[1] < max_y)
                & (ego_bbox[-1] >= 0)
            )
            if not valid_mask:
                continue
            bboxes.append(ego_bbox)
            velocity = gtbox[idx]['velocity']
            if np.isnan(velocity).any():
                velocity = [0.0, 0.0, 0.0]
            gt.append(gtbox[idx]['bbox'] + [gtbox[idx]['yaw']] + velocity[:2]) # gtbox[idx]['velocity'][:2]
            del box
        bboxes = bbox_ego2bev(bboxes, bev_size)
        if len(bboxes) == 0:
            bboxes = []
        else:
            bboxes = np.stack(bboxes)
        return bboxes, gt

    def get_ego_bboxes(self, bev_range):
        if bev_range is None:
            raise ValueError("bev_size is needed to gen bev bboxes")

        gt_bboxes = self.sample["gt_bboxes"]

        bboxes = []
        for bbox in gt_bboxes:
            if bbox["num_lidar_pts"] == 0 and bbox["num_radar_pts"] == 0:
                continue
            box = Bbox(self.sample, bbox)
            ego_bbox = box.get_bev_bboxes()
            valid_mask = (
                (ego_bbox[0] > bev_range[0])
                & (ego_bbox[0] < bev_range[3])
                & (ego_bbox[1] > bev_range[1])
                & (ego_bbox[1] < bev_range[4])
                & (ego_bbox[2] > bev_range[2])
                & (ego_bbox[2] < bev_range[5])
                & (ego_bbox[9] >= 0)
            )
            if not valid_mask:
                continue
            bboxes.append(ego_bbox)
            del box
        if len(bboxes) == 0:
            bboxes = []
        else:
            bboxes = np.stack(bboxes)
        return bboxes

    def get_lidar_bboxes(self, max_dist=55):
        gt_bboxes = self.sample["gt_bboxes"]

        bboxes = []
        instance_ids = []
        for bbox in gt_bboxes:
            if bbox["num_lidar_pts"] == 0 and bbox["num_radar_pts"] == 0:
                continue
            box = Bbox(self.sample, bbox)
            lidar_bbox = box.get_lidar_bboxes()
            dist = np.sqrt(np.sum(lidar_bbox[:2] ** 2))
            if dist > max_dist:
                continue
            bboxes.append(lidar_bbox)
            if "instance_id" in bbox:
                instance_ids.append(bbox["instance_id"])
            del box
        if len(bboxes) == 0:
            bboxes = []
            instance_ids = []
        else:
            bboxes = np.stack(bboxes)
        return bboxes, instance_ids

    def _get_map_info(self, vector_map, patch_size, canvas_size):
        vectors = vector_map.gen_vectorized_samples(
            self.sample["location"],
            self.sample["ego2global_translation"],
            self.sample["ego2global_rotation"],
        )

        for vector in vectors:
            pts = vector["pts"]
            vector["pts"] = np.concatenate(
                (pts, np.zeros((pts.shape[0], 1))), axis=1
            )

        for vector in vectors:
            vector["pts"] = vector["pts"][:, :2]

        (
            semantic_masks,
            instance_masks,
            forward_masks,
            backward_masks,
        ) = preprocess_map(vectors, patch_size, canvas_size, 3, 5, 36)
        self.sample["gt_mask"] = semantic_masks

    def get_bev_mask(
        self,
    ) -> np.array:
        """Get bev seg mask."""
        if "gt_mask" in self.sample:
            semantic_masks = self.sample["gt_mask"].astype(np.int64)
            num_cls = semantic_masks.shape[0]
            indices = np.arange(1, num_cls + 1).reshape(-1, 1, 1)
            semantic_indices = np.sum(semantic_masks * indices, axis=0)
            return semantic_masks, semantic_indices
        raise ValueError("No gt_mask")

    def _get_homography_by_cam(self, cam):
        '''return ego2img'''
        ego2sensor = np.array(cam['ego2camera'])

        camera_intrinsic = cam["camera_intrinsic"]
        camera_intrinsic = np.array(camera_intrinsic)

        viewpad = np.eye(4)
        viewpad[
            : camera_intrinsic.shape[0], : camera_intrinsic.shape[1]
        ] = camera_intrinsic
        ego2img = viewpad @ ego2sensor

        return ego2img

    def get_cam_by_name(self, name: str) -> Dict:
        """Get cam info by cam name."""
        cams = self.sample["cam"]
        cam = None
        for c in cams:
            if name == c["name"]:
                cam = copy.deepcopy(c)
        # print(cam['img_path'])
        if cam is None:
            raise ValueError(f"Cannot find cam {name}")
        cam["img"] = self._load_img(cam["img"])
        return cam

    def _get_global2img(self, ego2img, ego2global):
        global2ego = np.linalg.inv(ego2global)
        global2img = ego2img @ global2ego
        return global2img

    def get_lidar2ego(self):
        l2e_t = np.array(self.sample["sensor2ego_translation"])
        l2e_r = np.array(self.sample["sensor2ego_rotation"])
        l2e_m = np.zeros((4, 4), dtype=np.float64)
        l2e_m[:3, :3] = Quaternion(l2e_r).rotation_matrix
        l2e_m[:3, 3] = np.array(l2e_t)
        l2e_m[3, 3] = 1.0
        return l2e_m

    def get_lidar2img(self, cam_global2img, lidar2global):
        lidar2img = []
        for g2i in cam_global2img:
            lidar2img.append(g2i @ lidar2global)
        return lidar2img

    def get_lidar2global(self, lidar2ego, ego2global):
        return ego2global @ lidar2ego

    def get_ego2gloabl(self):
        e2g_t = np.array(self.sample["ego2global_translation"])
        e2g_r = np.array(self.sample["ego2global_rotation"])
        e2g_m = np.zeros((4, 4), dtype=np.float64)
        e2g_m[:3, :3] = Quaternion(e2g_r).rotation_matrix
        e2g_m[:3, 3] = np.array(e2g_t)
        e2g_m[3, 3] = 1.0

        return e2g_m

    def _get_cam2ego(self, cam):
        sensor2ego_translation = cam["sensor2ego_translation"]
        sensor2ego_rotation = cam["sensor2ego_rotation"]

        cam2ego = np.eye(4)
        rotation = Quaternion(sensor2ego_rotation).rotation_matrix
        cam2ego[:3, :3] = rotation
        cam2ego[:3, 3] = sensor2ego_translation
        return cam2ego

    def _get_cam_ego2global(self, cam):
        ego2global_translation = cam["ego2global_translation"]
        ego2global_rotation = cam["ego2global_rotation"]

        cam_ego2global = np.eye(4)
        rotation = Quaternion(ego2global_rotation).rotation_matrix
        cam_ego2global[:3, :3] = rotation
        cam_ego2global[:3, 3] = ego2global_translation
        return cam_ego2global

    def get_location(self):
        return self.sample["location"]

    def get_token(self):
        return self.sample["sample_token"]

    def get_meta(self):
        meta = {}

        meta["sample_token"] = self.sample['scene_token']

        # include = [
        #     "sample_token",
        #     "sensor2ego_rotation",
        #     "sensor2ego_translation",
        #     "ego2global_rotation",
        #     "ego2global_translation",
        #     "location",
        #     "scene",
        # ]
        # for k, v in self.sample.items():
        #     if k in include:
        #         meta[k] = v
        # meta["lidar2ego_translation"] = self.sample["sensor2ego_translation"]
        # meta["lidar2ego_rotation"] = self.sample["sensor2ego_rotation"]
        return meta

    def _load_img(self, img):
        # print(img[:50])
        img = decode_img(img, iscolor=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w, c = img.shape
        # print(h, w, c)
        return Image.fromarray(img)

    def get_center2d(self, bbox, cam):
        center3d = bbox[:3]
        camera_intrinsic = cam["camera_intrinsic"]
        camera_intrinsic = np.array(camera_intrinsic)
        center2d = camera_intrinsic @ center3d
        center2d[:2] = center2d[:2] / center2d[2]
        return center2d[:2], center2d[2]

    def get_corner2d(self, corners3d, cam, im_size):
        in_front = np.argwhere(corners3d[2, :] > 0).flatten()
        corners3d = corners3d[:, in_front]
        camera_intrinsic = cam["camera_intrinsic"]
        camera_intrinsic = np.array(camera_intrinsic)
        corner2d = camera_intrinsic @ corners3d
        corner2d = np.transpose(corner2d, (1, 0))
        corner2d[..., 0] = corner2d[..., 0] / corner2d[..., 2]
        corner2d[..., 1] = corner2d[..., 1] / corner2d[..., 2]

        if len(corner2d) == 0:
            min_x = 0
            min_y = 0
            max_x = 0
            max_y = 0
        else:
            min_x = min(corner2d[..., 0])
            min_y = min(corner2d[..., 1])
            max_x = max(corner2d[..., 0])
            max_y = max(corner2d[..., 1])

            min_x = max(min_x, 0)
            min_y = max(min_y, 0)
            max_x = min(max_x, im_size[0])
            max_y = min(max_y, im_size[1])

        return np.array([min_x, min_y, max_x, max_y])

    def get_mono_by_index(self, index):
        cam = self.sample["cam"][index]
        return self.get_mono_by_name(cam["name"])

    def get_mono_by_name(self, name):
        mono = {}
        cam = self.get_cam_by_name(name)
        im_size = cam["img"].size
        gt_bboxes = self.sample["gt_bboxes"]
        bboxes = []
        center2ds = []
        depths = []
        corner2ds = []
        attr_labels = []
        for bbox in gt_bboxes:
            box = Bbox(self.sample, bbox)
            cam_bbox = box.get_cam_bboxes(cam)
            corners3d = box.corners()
            center2d, depth = self.get_center2d(cam_bbox, cam)
            corner2d = self.get_corner2d(corners3d, cam, im_size)
            w = corner2d[2] - corner2d[0]
            h = corner2d[3] - corner2d[1]
            valid_mask = (
                (corner2d[0] < im_size[0])
                & (corner2d[2] > 0)
                & (corner2d[1] < im_size[1])
                & (corner2d[3] > 0)
                & (cam_bbox[9] >= 0)
                & (depth > 0)
                & (h > 1)
                & (w > 1)
            )
            if not valid_mask:
                continue
            center2ds.append(center2d)
            depths.append(depth)
            corner2ds.append(corner2d)
            bboxes.append(cam_bbox)
            attr_labels.append(box.get_attr_label())
        if len(center2ds) > 0:
            center2ds = np.stack(center2ds)
            depths = np.stack(depths)
            corner2ds = np.stack(corner2ds)
            bboxes = np.stack(bboxes)
        mono["img"] = cam["img"]
        mono["file_name"] = cam["img_path"]
        mono["gt_bboxes_3d"] = bboxes
        mono["center2d"] = center2ds
        mono["depth"] = depths
        mono["gt_bboxes"] = corner2ds
        mono["ego2global_translation"] = cam["ego2global_translation"]
        mono["ego2global_rotation"] = cam["ego2global_rotation"]

        mono["sensor2ego_translation"] = cam["sensor2ego_translation"]
        mono["sensor2ego_rotation"] = cam["sensor2ego_rotation"]
        mono["camera_intrinsic"] = cam["camera_intrinsic"]
        mono["attr_labels"] = attr_labels
        mono["token"] = self.get_token()
        return mono

    def _remove_close(self, points, radius=1.0):
        if isinstance(points, np.ndarray):
            points_numpy = points
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def get_lidar_points(
        self,
        num_sweeps,
        load_dim,
        use_dim,
        time_dim=4,
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=False,
    ):
        points = np.array(self.sample["lidar_points"], dtype=np.float32)
        ts = self.sample["timestamp"] / 1e6

        points = points.reshape(-1, load_dim)
        points[:, time_dim] = 0

        sweeps = self.sample["sweeps"]
        points_list = [points]

        if num_sweeps > 0:
            if pad_empty_sweeps and len(sweeps) == 0:
                for _ in range(num_sweeps):
                    if remove_close:
                        points_list.append(self._remove_close(points))
                    else:
                        points_list.append(points)
            else:
                if len(sweeps) <= num_sweeps:
                    choices = np.arange(len(sweeps))
                elif test_mode:
                    choices = np.arange(num_sweeps)
                else:
                    choices = np.random.choice(
                        len(sweeps), num_sweeps, replace=False
                    )
                for idx in choices:
                    sweep = sweeps[idx]
                    points_sweep = np.array(
                        sweep["lidar_points"], dtype=np.float32
                    )
                    points_sweep = points_sweep.reshape(-1, load_dim)
                    if remove_close:
                        points_sweep = self._remove_close(points_sweep)
                    sweep_ts = sweep["timestamp"] / 1e6
                    points_sweep[:, :3] = (
                        points_sweep[:, :3]
                        @ np.array(sweep["sensor2lidar_rotation"]).T
                    )
                    points_sweep[:, :3] += np.array(
                        sweep["sensor2lidar_translation"]
                    )
                    points_sweep[:, time_dim] = ts - sweep_ts
                    points_list.append(points_sweep)
            points = np.concatenate(points_list)
        points = points[:, use_dim]

        return points

    def get_lidar_ann_info(
        self, use_valid_flag=False, with_velocity=True, classes=None
    ):
        info = self.sample
        # filter out bbox containing no points
        if use_valid_flag:
            mask = np.array(info["valid_flag"], dtype=bool)
        else:
            mask = np.array(info["num_lidar_pts"]) > 0
        gt_bboxes_3d = np.array(info["gt_boxes"]).reshape(-1, 7)[mask]

        gt_names_3d = []
        for obj in info["gt_names"]:
            if isinstance(obj, bytes):
                obj = obj.decode("utf-8")
            gt_names_3d.append(obj)
        gt_names_3d = np.array(gt_names_3d)[mask]

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in classes:
                gt_labels_3d.append(classes.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if with_velocity:
            gt_velocity = np.array(info["gt_velocity"]).reshape(-1, 2)[mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        locs = gt_bboxes_3d[:, :3]
        dims = gt_bboxes_3d[:, 3:6]
        rots = gt_bboxes_3d[:, 6:7]
        vel = gt_bboxes_3d[:, 7:9]
        gt_bboxes_3d = np.concatenate(
            [locs, dims[:, [1, 0, 2]], vel, -rots - np.pi / 2], axis=-1
        )

        anns_results = dict(  # noqa C408
            boxes=gt_bboxes_3d,
            labels=gt_labels_3d,
            names=gt_names_3d,
        )
        return anns_results

    def get_lidar_points_with_seg(
        self,
        num_sweeps,
        load_dim,
        use_dim,
        time_dim=4,
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=False,
    ):
        points = np.array(self.sample["lidar_points"], dtype=np.float32)
        ts = self.sample["timestamp"] / 1e6

        points = points.reshape(-1, load_dim)
        points[:, time_dim] = 0

        points_label = np.array(
            self.sample["seg_gt_label"], dtype=np.uint8
        ).reshape([-1])
        points_label = np.vectorize(NUSCENES_SEMANTIC_MAPPING.__getitem__)(
            points_label
        )

        assert len(points) == len(
            points_label
        ), "points and segmentation labels do not match"

        if num_sweeps > 0:
            sweeps = self.sample["sweeps"]

            points_list = [points]

            if pad_empty_sweeps and len(sweeps) == 0:
                for _ in range(num_sweeps):
                    if remove_close:
                        points_list.append(self._remove_close(points))
                    else:
                        points_list.append(points)
            else:
                if len(sweeps) <= num_sweeps:
                    choices = np.arange(len(sweeps))
                elif test_mode:
                    choices = np.arange(num_sweeps)
                else:
                    choices = np.random.choice(
                        len(sweeps), num_sweeps, replace=False
                    )
                for idx in choices:
                    sweep = sweeps[idx]
                    points_sweep = np.array(
                        sweep["lidar_points"], dtype=np.float32
                    )
                    points_sweep = points_sweep.reshape(-1, load_dim)
                    if remove_close:
                        points_sweep = self._remove_close(points_sweep)
                    sweep_ts = sweep["timestamp"] / 1e6
                    points_sweep[:, :3] = (
                        points_sweep[:, :3]
                        @ np.array(sweep["sensor2lidar_rotation"]).T
                    )
                    points_sweep[:, :3] += np.array(
                        sweep["sensor2lidar_translation"]
                    )
                    points_sweep[:, time_dim] = ts - sweep_ts
                    points_list.append(points_sweep)
            points = np.concatenate(points_list)

        points = points[:, use_dim]

        return points, points_label

    def num_cams(self):
        return len(self.sample["cam"])

    def _get_ego_occ(self):
        voxel_semantics = np.frombuffer(
            self.sample["voxel_semantics"], dtype=np.uint8
        ).reshape(200, 200, 16)

        mask_lidar = np.frombuffer(
            self.sample["mask_lidar"], dtype=np.uint8
        ).reshape(200, 200, 16)

        mask_camera = np.frombuffer(
            self.sample["mask_camera"], dtype=np.uint8
        ).reshape(200, 200, 16)

        gt_occ = {
            "voxel_semantics": voxel_semantics,
            "mask_lidar": mask_lidar,
            "mask_camera": mask_camera,
        }
        return gt_occ

    def _get_occ_ego2lidar(
        self,
        voxel_semantics,
        mask_lidar,
        mask_camera,
        transform_matrix,
        pc_range=(-40, -40, -1, 40, 40, 5.4),
        lidar_pc_range=(-40.0, -40.0, -5.0, 40.0, 40.0, 3.0),
        ego_resolution=(0.4, 0.4, 0.4),
    ):
        H, W, Z = voxel_semantics.shape
        occ_trans = np.full((200, 200, 16), 17, dtype=np.uint8)
        mask_lidar_trans = np.full((200, 200, 16), 0, dtype=np.uint8)
        mask_camera_trans = np.full((200, 200, 16), 0, dtype=np.uint8)

        x = np.linspace(0.5, W - 0.5, W)
        y = np.linspace(0.5, H - 0.5, H)
        z = np.linspace(0.5, Z - 0.5, Z)
        xs, ys, zs = np.meshgrid(x / W, y / H, z / Z, indexing="ij")
        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        lidar_coords = np.stack([xs, ys, zs, np.ones_like(xs)], axis=-1)
        lidar_coords[..., 0:1] = (
            lidar_coords[..., 0:1] * (lidar_pc_range[3] - lidar_pc_range[0])
            + lidar_pc_range[0]
        )
        lidar_coords[..., 1:2] = (
            lidar_coords[..., 1:2] * (lidar_pc_range[4] - lidar_pc_range[1])
            + lidar_pc_range[1]
        )
        lidar_coords[..., 2:3] = (
            lidar_coords[..., 2:3] * (lidar_pc_range[5] - lidar_pc_range[2])
            + lidar_pc_range[2]
        )

        transformed_coords = lidar_coords @ transform_matrix.T

        x_ego, y_ego, z_ego = (
            transformed_coords[:, :, :, 0],
            transformed_coords[:, :, :, 1],
            transformed_coords[:, :, :, 2],
        )
        x_ego = ((x_ego - pc_range[0]) / ego_resolution[0]).astype(np.int32)
        y_ego = ((y_ego - pc_range[1]) / ego_resolution[1]).astype(np.int32)
        z_ego = ((z_ego - pc_range[2]) / ego_resolution[2]).astype(np.int32)

        valid_mask = (
            (x_ego >= 0)
            & (x_ego < 200)
            & (y_ego >= 0)
            & (y_ego < 200)
            & (z_ego >= 0)
            & (z_ego < 16)
        )

        x_lidar, y_lidar, z_lidar = (
            x[valid_mask].astype(np.int32),
            y[valid_mask].astype(np.int32),
            z[valid_mask].astype(np.int32),
        )
        x_ego, y_ego, z_ego = (
            x_ego[valid_mask],
            y_ego[valid_mask],
            z_ego[valid_mask],
        )
        occ_trans[x_lidar, y_lidar, z_lidar] = voxel_semantics[
            x_ego, y_ego, z_ego
        ]
        mask_lidar_trans[x_lidar, y_lidar, z_lidar] = mask_lidar[
            x_ego, y_ego, z_ego
        ]
        mask_camera_trans[x_lidar, y_lidar, z_lidar] = mask_camera[
            x_ego, y_ego, z_ego
        ]

        return occ_trans, mask_lidar_trans, mask_camera_trans

    def _get_lidar_occ(self, lidar2global=None):
        voxel_semantics = np.frombuffer(
            self.sample["voxel_semantics"], dtype=np.uint8
        ).reshape(200, 200, 16)
        mask_lidar = np.frombuffer(
            self.sample["mask_lidar"], dtype=np.uint8
        ).reshape(200, 200, 16)
        mask_camera = np.frombuffer(
            self.sample["mask_camera"], dtype=np.uint8
        ).reshape(200, 200, 16)

        cams = self.sample["cam"]
        cam_ego2global = self._get_cam_ego2global(cams[0])
        global2camego = np.linalg.inv(cam_ego2global)
        if lidar2global is None:
            lidar2ego = self.get_lidar2ego()
            ego2global = self.get_ego2gloabl()
            lidar2global = self.get_lidar2global(lidar2ego, ego2global)

        lidar2camego = global2camego @ lidar2global

        (
            lidar_voxel_semantics,
            mask_lidar,
            mask_camera,
        ) = self._get_occ_ego2lidar(
            voxel_semantics, mask_lidar, mask_camera, lidar2camego
        )

        gt_occ = {
            "ego_voxel_semantics": voxel_semantics,
            "voxel_semantics": lidar_voxel_semantics,
            "mask_lidar": mask_lidar,
            "mask_camera": mask_camera,
            "lidar2camego": lidar2camego,
        }

        return gt_occ

class NuscenesBevSampler(object):
    """Bev Dataset object for packed NuScenes.

    Args:
        with_bev_bboxes: Whether include bev bboxes.
        with_bev_mask: Whether include bev bboxes.
        map_path: Path to Nuscenes Map, needed if include bev mask.
        line_classes: Classes of line. ex. road divider, lane divider.
        ped_crossing_classes: Classes of ped corssing. ex. ped_crossing
        contour_classes: Classes of contour. ex. road segment, lane.
        bev_size: Size for bev using meter. ex. (51.2, 51.2, 0.2)
        bev_range: range for bev, alternative of bev_size.
            ex.(-61.2, -61.2, -2, 61.2, 61.2, 10)
        map_size: Size for seg map.
        need_lidar: Whether need lidar points. Default: False.
        num_sweeps: Number of sweeps, if lidar points is needed.
                    Default: 0

        load_dim: Dimension number of the loaded points.
                  Defaults to 5.
        use_dim: Which dimension to use.
        need_mono_data: Whether need mono data. Default: False.
        with_ego_occ: Whether include ego occ. Default: False.
    """

    def __init__(
        self,
        with_bev_bboxes: bool = True,
        with_ego_bboxes: bool = False,
        with_bev_mask: bool = True,
        with_lidar_bboxes: bool = False,
        max_dist: Optional[float] = 55.0,
        map_path: Optional[str] = None,
        line_classes: Optional[List[str]] = None,
        ped_crossing_classes: Optional[List[str]] = None,
        contour_classes: Optional[List[str]] = None,
        bev_size: Optional[Tuple] = None,
        bev_range: Optional[Tuple] = None,
        map_size: Optional[Tuple] = None,
        need_lidar: bool = False,
        num_sweeps: Optional[int] = 0,
        load_dim: Optional[int] = 5,
        use_dim: Optional[List[int]] = None,
        need_mono_data: bool = False,
        with_ego_occ: bool = False,
        **kwargs,
    ):
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.map_path = map_path
        self.map_size = map_size

        self.with_bev_mask = with_bev_mask
        self.with_bev_bboxes = with_bev_bboxes
        self.with_ego_bboxes = with_ego_bboxes
        self.with_lidar_bboxes = with_lidar_bboxes
        self.need_lidar = need_lidar
        self.need_mono_data = need_mono_data
        self.max_dist = max_dist
        self.num_sweeps = num_sweeps
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.with_ego_occ = with_ego_occ

        if use_dim is None:
            self.use_dim = [0, 1, 2]

        if self.with_bev_mask is True:
            self.patch_size = (
                self.map_size[0] * 2,
                self.map_size[1] * 2,
            )
            self.canvas_size = (
                int(self.map_size[0] * 2 / self.map_size[2]),
                int(self.map_size[1] * 2 / self.map_size[2]),
            )

            if line_classes is None:
                line_classes = ["road_divider", "lane_divider"]
            if ped_crossing_classes is None:
                ped_crossing_classes = ["ped_crossing"]
            if contour_classes is None:
                contour_classes = ["road_segment", "lane"]
            self.vector_map = VectorizedLocalMap(
                map_path,
                self.patch_size,
                self.canvas_size,
                line_classes=line_classes,
                ped_crossing_classes=ped_crossing_classes,
                contour_classes=contour_classes,
            )

    def __call__(self, sample):
        # print("__call__(self, sample)")
        data = {}
        if self.map_path and self.with_bev_mask:
            sample._get_map_info(
                self.vector_map, self.patch_size, self.canvas_size
            )

        instance_ids = []
        if self.with_lidar_bboxes:
            (
                data["lidar_bboxes_labels"],
                instance_ids,
            ) = sample.get_lidar_bboxes(max_dist=self.max_dist)
        gt = None
        # print("sample.get_bboxes_by_bev(self.bev_size)")
        if self.with_bev_bboxes:
            data["bev_bboxes_labels"], gt = sample.get_bboxes_by_bev(self.bev_size)
            # print("sample.get_bboxes_by_bev(self.bev_size) end")
        if self.with_ego_bboxes:
            data["ego_bboxes_labels"] = sample.get_ego_bboxes(self.bev_range)
        if self.with_bev_mask:
            (
                data["bev_seg_mask"],
                data["bev_seg_indices"],
            ) = sample.get_bev_mask()

        cam_names = [
            "fisheye_camera_front",
            "fisheye_camera_left",
            "fisheye_camera_rear",
            "fisheye_camera_right"
        ]
        imgs = []
        lidar2cam = []
        img_names = []
        lidar2img = []
        cam2lidar = []
        for name in cam_names:
            # print(name)
            cam = sample.get_cam_by_name(name)
            imgs.append(cam["img"])
            # cam['img'].sae(r"home/Desktop/fisheyedod/horizon/infer_out/tmp.jpg")
            img_names.append(cam["img_path"])
            cam2lidar.append(np.array(cam['cam2lidar']))
            lidar2cam.append(np.array(cam['lidar2cam']))
            lidar2img.append(np.array(cam['lidar2img']))
            # print(222222222, name, cam["img_path"], cam_names)
        gt_box = np.array(gt)
        if len(gt) > 0:
            data['gt_label'] = data['bev_bboxes_labels'][:, -1].astype(np.int32)
        else:
            data['gt_label'] = np.array([])
        data['gt_boxes_3d'] = gt_box
        # tmp2 = np.array(bbox_bev2ego(data['bev_bboxes_labels'], self.bev_size))
        # kk = np.min(tmp2[:, :7] - data['gt_boxes_3d'][:, :7])
        # tmp = kk.all()
        data["img"] = imgs
        # cv2.imwrte(r"home/Desktop/fisheyedod/horizon/infer_out/tmp.jpg", data['img'][0])
        data["img_name"] = img_names
        data["cam_name"] = cam_names
        data["cam2lidar"] = np.array(cam2lidar)
        data["lidar2cam"] = np.array(lidar2cam)
        data["lidar2img"] = np.array(lidar2img)
        # data["camera_intrinsic"] = "omni_cam"
        data["layout"] = "hwc"
        data["color_space"] = "rgb"
        # data["sample_token"] = sample.sample['scene_token']
        names = img_names[0].split(os.sep)
        data["sample_token"] = ".".join([names[-4], names[-3], names[-1]])
        data["instance_ids"] = instance_ids
        if self.need_lidar:
            data["points"] = sample.get_lidar_points(
                num_sweeps=self.num_sweeps,
                load_dim=self.load_dim,
                use_dim=self.use_dim,
            )

        if self.need_mono_data:
            center2ds = []
            corner2ds = []
            mono_3d_bboxes = []
            depths = []
            attr_labels = []
            gt_labels_3ds = []
            for _, name in enumerate(cam_names):
                mono = sample.get_mono_by_name(name)
                center2ds.append(np.array(mono["center2d"]))
                corner2ds.append(np.array(mono["gt_bboxes"]))
                gt_bboxes_3d = np.array(mono["gt_bboxes_3d"]).reshape((-1, 10))
                gt_labels_3ds.append(torch.from_numpy(gt_bboxes_3d[:, -1]))
                gt_bboxes_3d = torch.from_numpy(gt_bboxes_3d[:, :-1])
                mono_3d_bboxes.append(
                    CameraInstance3DBoxes(
                        gt_bboxes_3d,
                        box_dim=gt_bboxes_3d.shape[-1],
                        origin=(0.5, 0.5, 0.5),
                    )
                )
                depths.append(np.array(mono["depth"]))
                attr_labels.append(mono["attr_labels"])
            data["center2ds"] = center2ds
            data["corner2ds"] = corner2ds
            data["mono_3d_bboxes"] = mono_3d_bboxes
            data["mono_3d_labels"] = gt_labels_3ds
            data["depths"] = depths
            data["attr_labels"] = attr_labels

        if self.with_ego_occ:
            data["gt_occ_info"] = sample._get_ego_occ()
        # print("data[] = sample.get_meta()")
        data["meta"] = sample.get_meta()
        # print("data[] = sample.get_meta() end")
        return data

class Fcos3dGOYUDataset(Dataset):
    """Dataset object for packed NuScenes.

    Args:
        data_path: packed dataset path.
        transforms: A function transform that takes input
            sample and its target as entry and returns a transformed version.
        pack_type: The pack type.
        pack_kwargs: Kwargs for pack type.
    """

    @require_packages(
        "nuscenes", raise_msg="Please `pip3 install nuscenes-devkit`"
    )
    def __init__(
        self,
        data_path: str,
        transforms: Optional[Callable] = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
        num_split: int = 1,
    ):
        self.root = data_path
        self.transforms = transforms

        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        try:
            self.pack_type = get_packtype_from_path(data_path)
        except NotImplementedError:
            assert pack_type is not None
            self.pack_type = PackTypeMapper(pack_type.lower())

        self.pack_file = self.pack_type(
            self.root, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()
        self.meta = self._decode_meta(self.pack_file)
        if self.meta is None:
            logger.warning("Packer data donot include meta. ")  # noqa [E501]
        if self.meta:
            org_len = len(self.meta["scene_info"])

            if num_split > 1:
                self._split_scenes(num_split)
                new_len = len(self.meta["scene_info"])
                # print(f"Split scenes from {org_len} to {new_len}")
            self._gen_idx2scenes()

    def _gen_idx2scenes(self):
        idx = 0
        self.idx2scenes = {}
        for k, v in self.meta["scene_info"].items():
            for _ in range(v):
                self.idx2scenes[idx] = k
                idx += 1
        # print(self.idx2scenes)

    def get_aug(self):
        aug_cfg = {}
        if self.transforms is not None:
            for trans in self.transforms.transforms:
                if hasattr(trans, "get_aug"):
                    name, cfg = trans.get_aug()
                    aug_cfg[name] = cfg
        return aug_cfg

    def set_aug(self, scene, aug_cfg):
        self.scenes_aug[scene] = aug_cfg

    def _split_scenes(self, num_split):
        new_scene_info = {}
        for k, v in self.meta["scene_info"].items():
            sequence_length = np.array(
                list(
                    range(
                        0,
                        v,
                        math.ceil(v / num_split),
                    )
                )
                + [v]
            )
            for idx, sub_seq_len in enumerate(
                sequence_length[1:] - sequence_length[:-1]
            ):
                name = k + f"_{idx}"
                new_scene_info[name] = sub_seq_len

        self.meta["scene_info"] = new_scene_info

    def get_meta(self):
        return self.meta

    def _decode_meta(self, pack_file):
        meta = self._decode(pack_file, "__meta__")
        return meta

    def _decode(self, pack_file, sample):
        def _decode_hook(obj):
            def _decode_bytes(obj):
                if isinstance(obj, bytes):
                    obj = obj.decode("utf-8")
                return obj

            new_obj = {}
            for k, v in obj.items():
                k = _decode_bytes(k)
                if k not in [
                    "img",
                    "voxel_semantics",
                    "mask_lidar",
                    "mask_camera",
                ]:
                    v = _decode_bytes(v)
                new_obj[k] = v
            return new_obj

        sample = pack_file.read(sample)
        if sample is not None:
            sample = msgpack.unpackb(
                sample, object_hook=_decode_hook, raw=True
            )
        return sample

    def __getstate__(self):
        state = self.__dict__
        state["pack_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.pack_file = self.pack_type(
            self.root, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()


@OBJECT_REGISTRY.register
class FisheyeGOYUMonoDataset(Fcos3dGOYUDataset):
    def __init__(self, **kwargs):
        super(FisheyeGOYUMonoDataset, self).__init__(**kwargs)
        sample = self._decode(self.pack_file, self.samples[0])
        sample = Fcos3dGOYUSample(sample)
        self.num_cams = sample.num_cams()

    def __len__(self):
        return len(self.samples) * self.num_cams

    def __getitem__(self, item):
        index = item // self.num_cams
        cam_idx = item % self.num_cams

        sample = self._decode(self.pack_file, self.samples[index])
        sample = Fcos3dGOYUSample(sample)
        data = sample.get_mono_by_index(cam_idx)
        data["img"] = np.array(data["img"])
        data["scale_factor"] = 1.0
        data["gt_bboxes_3d"] = np.array(data["gt_bboxes_3d"]).reshape((-1, 10))
        data["gt_labels_3d"] = torch.from_numpy(data["gt_bboxes_3d"][:, -1])
        data["gt_bboxes_3d"] = torch.from_numpy(data["gt_bboxes_3d"][:, :-1])
        data["gt_bboxes_3d"] = CameraInstance3DBoxes(
            data["gt_bboxes_3d"],
            box_dim=data["gt_bboxes_3d"].shape[-1],
            origin=(0.5, 0.5, 0.5),
        )
        data["gt_labels"] = data["gt_labels_3d"]
        data["depths"] = torch.from_numpy(np.array(data.pop("depth")))
        data["cam2img"] = data.pop("camera_intrinsic")
        data["centers2d"] = torch.from_numpy(np.array(data.pop("center2d")))
        data["attr_labels"] = torch.tensor(data["attr_labels"])
        data["gt_bboxes"] = torch.from_numpy(np.array(data["gt_bboxes"]))
        data["layout"] = "hwc"
        data["color_space"] = "rgb"
        data["filename"] = data["file_name"]
        if self.transforms is not None:
            data = self.transforms(data)

        return data

class rotateTranslation(object):
    def __init__(self, roll, pitch, yaw, x=0, y=0, z=0, rowMajor = True, inv = False):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.x = x
        self.y = y
        self.z = z
        self.rowMajor = rowMajor
        self.inv = inv
        self.allType = ["ZYX", "ZYZ", "ZXY", "ZXZ", "YXY", "YZX", "YXZ", "YZY", "XYX", "XYZ", "XZX", "XZY"]
    
    def Xrotate(self, angle):
        return np.array([[1, 0, 0], \
                                [0, np.cos(angle), -np.sin(angle)], \
                                [0, np.sin(angle), np.cos(angle)]], \
                                dtype=np.float64)
    def Yrotate(self, angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], \
                                 [0, 1, 0], \
                                 [-np.sin(angle), 0, np.cos(angle)]], \
                                dtype=np.float64)
    def Zrotate(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], \
                               [np.sin(angle), np.cos(angle), 0], \
                               [0, 0, 1]], \
                                dtype=np.float64)

    def postprocess(self, ret, flatten = True):
        ones = np.zeros((4, 4), dtype=np.float64)

        ones[:3, :3] = ret
        ones[0, 3] = self.x
        ones[1, 3] = self.y
        ones[2, 3] = self.z
        ones[3, 3] = 1
        if self.inv:
            ones = np.linalg.inv(ones)
        if flatten:
            if self.rowMajor:
                return list(ones.flatten().tolist())
            else:
                return list(ones.T.flatten().tolist())
        return ones

    def getRotateZYX(self, flatten = False):
        ret = np.matmul(self.Xrotate(self.roll), np.matmul(self.Yrotate(self.pitch), self.Zrotate(self.yaw)))
        return self.postprocess(ret, flatten = flatten)

class FisheyeParser(object):
    """Parser object for packed NuScenes.

    Args:
        version: Version for nuscenes.
        src_data_dir: Path for data.
        split_name: Split_name for dataset.(ex. "train", "val")
        max_sweeps: Max number of sweeps for packed lidar points.
        need_occ: Whether parse occ data.
    """
    def __init__(
        self,
        src_data_dir: str,
        meta_json_dir: str,
        exclude_dirs: str,
        split_name: str = "val",
        max_sweeps: int = 10,
        need_occ: bool = False,
        camera_type: list = [],
        choose_camera: str = "",
    ):
        self.src_data_dir = src_data_dir
        self.choose_camera = choose_camera

        infoPath = os.path.join(meta_json_dir, f"fisheye3dod_infos_{split_name}.pkl")
        self.calib_all = {}
        jsonfile = mmengine.load(infoPath)
        data_list = jsonfile['data_list']
        self.category = jsonfile['metainfo']['categories']
        self.classes = list(jsonfile['metainfo']['categories'].keys())

        dic = {}
        for idx, anno in enumerate(data_list):
            data_list[idx]['cam_info'].pop('cam_nusc')
            scene = data_list[idx]['token'].split('.ego')[0]
            if scene not in dic.keys():
                dic[scene] = 1
            else:
                dic[scene] += 1
        self.samples = data_list
        self.scene2len = dic

    def _get_meta(self):
        meta = {"scene_info": self.scene2len}
        return meta

    def _gen_info(self, sample_token):
        # sample_token： self.samples, element
        sample_info = {}
        sample_info["scene_token"] = sample_token['token'].split('.ego')[0]
        self._gen_cam_info(sample_token, sample_info)
        self._gen_anns(sample_token, sample_info)
        # self._gen_lidar_info(sample_token, sample_info)
        '''
        gt_bboxes_list=data["gt_bboxes"], #list [...] torch.Size([47, 4]), torch.Size([5, 4]),  torch.Size([2, 4])
        gt_labels_list=data["gt_labels"], # list [...] torch.Size([47]), torch.Size([5]) torch.Size([2])
        gt_bboxes_3d_list=data["gt_bboxes_3d"], # list CameraInstance3DBoxes, tensor:torch.Size([47, 9]) torch.Size([5, 9])
        gt_labels_3d_list=data["gt_labels_3d"], # list torch.Size([47]), torch.Size([5]), torch.Size([2]), same as labels
        centers2d_list=data["centers2d"], # list torch.Size([47, 2])  torch.Size([5, 2])  torch.Size([2, 2])
        depths_list=data["depths"], # list torch.Size([47]), torch.Size([5]), torch.Size([2])
        attr_labels_list=data["attr_labels"], # list torch.Size([47]), torch.Size([5]), torch.Size([2])
        input_meta["box_type_3d"]
        input_meta["cam2img"]
        '''
        return sample_info

    def _gen_anns(self, sample, info):
        gt_bboxes = []
        for i in range(len(sample['annos']['gt_names'])):
            box = sample['annos']['gt_boxes'][i]
            center = np.array([ box[0], box[1], box[2] ])
            size_lwh = np.array([ box[3], box[4], box[5] ])
            wlh = np.array(size_lwh)[[1, 0, 2]] # lwh -> w, l, h
            yaw = float(box[6])
            quat = Quaternion(axis=[0, 0, 1], radians=yaw)
            rot = quat.elements.tolist()

            gt_bbox = {
                "bbox": np.concatenate([center, wlh], axis=0).tolist(),
                "rot": rot,
                "attr_name": "None",
                "attr_id": -1,
                "cat": sample['annos']['gt_names'][i],
                "token": sample['token'],
                "velocity": [np.nan, np.nan, np.nan],
                "yaw":yaw,
            }
            gt_bboxes.append(gt_bbox)
        info["gt_bboxes"] = gt_bboxes

        boxes = []
        categories = []
        for i in range(len(sample['annos']['gt_names'])):
            box = sample['annos']['gt_boxes'][i]
            center = np.array([ box[0], box[1], box[2] ])
            size_lwh = np.array([ box[3], box[4], box[5] ])
            yaw = float(box[6])
            boxes.append(np.array([center[0], center[1], center[2], size_lwh[0], size_lwh[1], \
                              size_lwh[2], yaw]))
            categories.append(sample['annos']['gt_names'][i])

        if len(boxes) > 0:
            boxes = np.stack(boxes)
            info['gt_boxes'] = boxes.tolist()
            info['gt_names'] = categories
            vel = np.ones_like(boxes[:, :3])
            vel[...] = np.nan
            info["gt_velocity"] = vel.tolist()
        else:
            info['gt_boxes'] = []
            info['gt_names'] = []
            info["gt_velocity"] = []

    def _load_img(self, img_path):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img = np.array(img)
        return cv2.imencode(".jpg", img)[1].tobytes(), w, h

    def _gen_cam_info(self, sample, info):
        scene_token = sample['token'].split('.ego')[0]
        cams = []
        for camType, caminfo in sample['cam_info']['cam_fisheye'].items():
            imgpath = os.path.join(self.src_data_dir, caminfo['cam_path'])
            image, w, h = self._load_img(imgpath)
            cam = {
                "img_path": imgpath,
                "img": image,
                "name": camType,
                "token": caminfo['cam_path'],
                "camera_intrinsic": -1,
                # "cam2img": caminfo['cam2img'].tolist(),
                "cam2lidar": caminfo['cam2lidar'].tolist(),
                "lidar2cam": caminfo['lidar2cam'].tolist(),
                "lidar2img": caminfo['lidar2img'].tolist(),
            }
            cams.append(cam)

        info["cam"] = cams

    def _read_points(self, lidar_path):
        lidar_file = os.path.join(self.root_path, lidar_path)
        points = np.fromfile(lidar_file, dtype=np.float32)
        return points

    def _gen_lidar_info(self, sample, info):
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self._gen_info(self.samples[index])

    @property
    def num_available_scenes(self):
        return len(self.scenes)

class BoxVisualize(NuScenesBox):
    """ Simple data class representing a 3d box including, label, score and velocity. """
    def __init__(self, center: list[float], size: list[float],
                 orientation: Quaternion, label=np.nan,
                 score=np.nan, velocity=np.nan, name=None, token=None, 
                 intrinsic = None, width = -1, height = -1):
        super(BoxVisualize, self).__init__(center,
                            size,
                            orientation,
                            label,
                            score,
                            velocity,
                            name,
                            token)
        self.intrinsic = intrinsic
        self.width = width
        self.height = height

    def get_corners(self,
                   view: np.ndarray = np.eye(3),
                   normalize: bool = False) -> None:
        """
        Renders box using OpenCV2.
        :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
        :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
        :param linewidth: Linewidth for plot.
        """
        corners = view_points(self.corners(), view, normalize=normalize)

        corners = np.matmul(self.intrinsic, corners)
        corners[0, :] = corners[0, :] / corners[2, :]
        corners[1, :] = corners[1, :] / corners[2, :]

        mask = (corners[0, :] < self.width) & (corners[0, :] >= 0)
        mask = mask & (corners[1, :] < self.height) & (corners[1, :] >= 0)
        mask = mask & (corners[2, :] >= 0)
        remain = corners[:, mask]
        return remain

class FisheyeImageParser(FisheyeParser):
    """Parser object for packed NuScenes images."""

    def _gen_sensor_info(self, sample, info):
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = self.nusc.get("sample_data", lidar_token)

        cs_rec = self.nusc.get(
            "calibrated_sensor", sd_rec["calibrated_sensor_token"]
        )
        pose_rec = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])

        info["sensor2ego_translation"] = cs_rec["translation"]
        info["sensor2ego_rotation"] = cs_rec["rotation"]
        info["ego2global_translation"] = pose_rec["translation"]
        info["ego2global_rotation"] = pose_rec["rotation"]


class fisheyeGOYUPacker(Packer):
    def __init__(
        self,
        target_data_dir: str,
        meta_json_dir: str,
        exclude_dirs: list,
        src_data_dir: str,
        num_workers: int,
        pack_type: str,
        only_lidar: bool = False,
        need_occ: bool = False,
        split_name: str = "train",
        camera_type: list = [],
        choose_camera: str = "",
        **kwargs,
    ):
        if only_lidar:
            logger.info("only process lidar samples ...")
            # self.dataset = Fcos3dGOYULidarParser(
            #     src_data_dir=src_data_dir,
            #     split_name=split_name,
            #     use_lidar_seg=False,
            # )
            # self.packer_meta = None
            exit(-1)
        else:
            self.dataset = FisheyeImageParser(
                meta_json_dir = meta_json_dir,
                exclude_dirs = exclude_dirs,
                src_data_dir = src_data_dir,
                need_occ = need_occ,
                split_name = split_name,
                camera_type = camera_type,
                choose_camera = choose_camera,
            )
            self.packer_meta = self.dataset._get_meta()
        num_samples = len(self.dataset)
        logger.info(
            f"Packing {split_name} datasets\
, total samples {num_samples}"
        )
        super(fisheyeGOYUPacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def _write_length(self, num):

        """
        Write max length to target format file.

        Args:
            num (int): Max length for packing data.
        """

        try:
            self.pack_file.write("__len__", "{}".format(num).encode("ascii"))
            if self.packer_meta is not None:
                self.pack_file.write(
                    "__meta__", msgpack.packb(self.packer_meta)
                )
        except ValueError:
            logger.warning(
                f"{self.target_pack_type} is not supported "
                + "for write length mainly due to the string key.",
            )
        self.pack_file.close()

    def pack_data(self, idx):
        data_meta = self.dataset[idx]
        return msgpack.packb(data_meta)


@OBJECT_REGISTRY.register
class FisheyeGOYUFromImage(Dataset):
    """Read NuScenes from image.

    Args:
        version: Version for nuscenes.
        src_data_dir: Path for data.
        split_name: Split_name for dataset.(ex. "train", "val")
        with_bev_bboxes: Whether include bev bboxes.
        with_bev_mask: Whether include bev bboxes.
        map_path: Path to Fcos3dGOYU Map, needed if include bev mask.
        line_classes: Classes of line. ex. road divider, lane divider.
        ped_crossing_classes: Classes of ped corssing. ex. ped_crossing
        contour_classes: Classes of contour. ex. road segment, lane.
        bev_size: Size for bev using meter. ex. (51.2, 51.2, 0.2)
        bev_range: range for bev, alternative of bev_size.
                   ex.(-61.2, -61.2, -2, 61.2, 61.2, 10)
        map_size: Size for seg map.
        need_lidar: Whether need lidar points. Default: False.
        num_sweeps: Number of sweeps, if lidar points is needed.
        load_dim: Dimension number of the loaded points.
        use_dim: Which dimension to use.
        with_ego_occ: Whether include ego occ.
        with_lidar_occ: Whether include lidar occ.
    """

    def __init__(
        self,
        version,
        src_data_dir,
        split_name="train",
        transforms=None,
        with_bev_bboxes: bool = True,
        with_ego_bboxes: bool = False,
        with_lidar_bboxes: bool = False,
        with_bev_mask: bool = True,
        map_path: Optional[str] = None,
        line_classes: Optional[List[str]] = None,
        ped_crossing_classes: Optional[List[str]] = None,
        contour_classes: Optional[List[str]] = None,
        bev_size: Optional[Tuple] = None,
        bev_range: Optional[Tuple] = None,
        map_size: Optional[Tuple] = None,
        need_lidar: bool = False,
        num_sweeps: Optional[int] = 0,
        load_dim: Optional[int] = 5,
        use_dim: Optional[List[int]] = None,
        with_ego_occ: bool = False,
        with_lidar_occ: bool = False,
    ):
        if with_ego_occ or with_lidar_occ:
            need_occ = True
        else:
            need_occ = False
        self.dataset = FisheyeImageParser(
            version=version,
            src_data_dir=src_data_dir,
            split_name=split_name,
            need_occ=need_occ,
        )
        self.sampler = Fcos3dGOYUBevSampler(
            with_bev_bboxes=with_bev_bboxes,
            with_ego_bboxes=with_ego_bboxes,
            with_lidar_bboxes=with_lidar_bboxes,
            with_bev_mask=with_bev_mask,
            map_path=map_path,
            line_classes=line_classes,
            ped_crossing_classes=ped_crossing_classes,
            contour_classes=contour_classes,
            bev_size=bev_size,
            bev_range=bev_range,
            map_size=map_size,
            need_lidar=need_lidar,
            num_sweeps=num_sweeps,
            load_dim=load_dim,
            use_dim=use_dim,
            with_ego_occ=with_ego_occ,
        )
        self.transforms = transforms
        self.with_lidar_occ = with_lidar_occ

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = Fcos3dGOYUSample(self.dataset[index])
        data = self.sampler(sample)
        if self.transforms is not None:
            data = self.transforms(data)
        if self.with_lidar_occ:
            data["gt_occ_info"] = sample._get_lidar_occ(data["lidar2global"])
        return data


@OBJECT_REGISTRY.register
class FisheyeGOYUFromImageSequence(FisheyeGOYUFromImage):
    def __init__(self, num_seq, **kwargs):
        super(FisheyeGOYUFromImageSequence, self).__init__(**kwargs)
        self.num_seq = num_seq

    def __getitem__(self, item):
        start = item
        end = item - self.num_seq
        seq_data = []

        for i in range(start, end, -1):
            idx = max(i, 0)
            data = super().__getitem__(idx)
            seq_data.append(data)
        return seq_data


@OBJECT_REGISTRY.register
class FisheyeGOYUMonoFromImage(Dataset):
    def __init__(
        self,
        version,
        src_data_dir,
        split_name="val",
        transforms=None,
    ):
        self.src_data_dir = src_data_dir
        self.transforms = transforms
        self.dataset = FisheyeImageParser(
            version=version,
            src_data_dir=src_data_dir,
            split_name=split_name,
        )
        sample = self.dataset[0]
        sample = Fcos3dGOYUSample(sample)
        self.num_cams = sample.num_cams()

    def __len__(self):
        return len(self.dataset) * self.num_cams

    def __getitem__(self, item: int):
        index = item // self.num_cams
        cam_idx = item % self.num_cams
        sample = self.dataset[index]
        sample = Fcos3dGOYUSample(sample)
        data = sample.get_mono_by_index(cam_idx)
        data["scale_factor"] = 1.0
        data["gt_bboxes_3d"] = np.array(data["gt_bboxes_3d"]).reshape((-1, 10))
        data["gt_labels_3d"] = torch.from_numpy(data["gt_bboxes_3d"][:, -1])
        data["gt_bboxes_3d"] = torch.from_numpy(data["gt_bboxes_3d"][:, :-1])
        data["gt_bboxes_3d"] = CameraInstance3DBoxes(
            data["gt_bboxes_3d"],
            box_dim=data["gt_bboxes_3d"].shape[-1],
            origin=(0.5, 0.5, 0.5),
        )
        data["gt_labels"] = data["gt_labels_3d"]
        data["depths"] = torch.from_numpy(np.array(data.pop("depth")))
        data["cam2img"] = data.pop("camera_intrinsic")
        data["centers2d"] = torch.from_numpy(np.array(data.pop("center2d")))
        data["attr_labels"] = torch.tensor(data["attr_labels"])
        data["gt_bboxes"] = torch.from_numpy(np.array(data["gt_bboxes"]))
        data["layout"] = "hwc"
        data["color_space"] = "rgb"
        data["filename"] = data["file_name"]
        image = Image.open(os.path.join(self.src_data_dir, data["filename"]))
        image = np.array(image)
        data["img"] = image
        if self.transforms is not None:
            data = self.transforms(data)
        return data


def create_nuscenes_groundtruth_database(
    dataset_class_name: str,
    dataset_path: str,
    info_prefix: str,
    out_dir: str,
    used_classes: Optional[List[str]] = None,
    database_save_path: Optional[str] = None,
    db_info_save_path: Optional[str] = None,
):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name: Name of the input dataset.
        dataset_path: Path of the dataset.
        info_prefix: Prefix of the info file.
        out_dir: Path of the output db file.
        used_classes: Classes have been used.
            Default: None.
        database_save_path: Path to save database.
            Default: None.
        db_info_save_path: Path to save db_info.
            Default: None.
    """
    # print(f"Create GT Database of {dataset_class_name}")
    dataset_path = osp.join(dataset_path, "train_lmdb")

    # if dataset_class_name == "Fcos3dGOYULidarDataset":
    #     dataset = Fcos3dGOYULidarDataset(
    #         data_path=dataset_path,
    #         num_sweeps=10,
    #         use_valid_flag=True,
    #         load_dim=5,
    #         use_dim=[0, 1, 2, 3, 4],
    #         pad_empty_sweeps=True,
    #         remove_close=True,
    #         classes=[
    #             "car",
    #             "truck",
    #             "construction_vehicle",
    #             "bus",
    #             "trailer",
    #             "barrier",
    #             "motorcycle",
    #             "bicycle",
    #             "pedestrian",
    #             "traffic_cone",
    #         ],
    #     )

    if database_save_path is None:
        database_save_path = osp.join(out_dir, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(
            out_dir, f"{info_prefix}_dbinfos_train.pkl"
        )
    dir_name = osp.expanduser(database_save_path)
    os.makedirs(dir_name, exist_ok=True)
    all_db_infos = {}

    group_counter = 0
    for j in tqdm(range(len(dataset))):
        example = dataset.prepare_data(j)
        annos = example["lidar"]["annotations"]
        image_idx = example["metadata"]["sample_token"]
        points = example["lidar"]["points"]
        gt_boxes_3d = annos["boxes"]
        names = annos["names"]
        group_dict = {}
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes_3d.shape[0]
        if num_obj == 0:
            continue
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f"{info_prefix}_gt_database", filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            with open(abs_filepath, "w") as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    "name": names[i],
                    "path": rel_filepath,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)


def create_nuscenes_infos(
    dataset_class_name: str,
    dataset_path: str,
    info_prefix: str,
    out_dir: str,
    info_save_path: Optional[str] = None,
):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name: Name of the input dataset.
        dataset_path: Path of the dataset.
        info_prefix: Prefix of the info file.
        out_dir: Path of the output info file.
        info_save_path: Path to save info file.
            Default: None.
    """
    print(f"Generate infos of {dataset_class_name}")
    dataset_path = osp.join(dataset_path, "train_lmdb")

    if dataset_class_name == "Fcos3dGOYULidarDataset":
        dataset = Fcos3dGOYULidarDataset(
            data_path=dataset_path,
            num_sweeps=10,
            use_valid_flag=True,
            load_dim=5,
            use_dim=[0, 1, 2, 3, 4],
            pad_empty_sweeps=True,
            remove_close=True,
            classes=[
                "car",
                "truck",
                "construction_vehicle",
                "bus",
                "trailer",
                "barrier",
                "motorcycle",
                "bicycle",
                "pedestrian",
                "traffic_cone",
            ],
        )

    os.makedirs(out_dir, exist_ok=True)

    nusc_infos = []
    for i in tqdm(range(len(dataset))):
        info = dataset.get_sample(i)
        new_info = {"sample_token": info["sample_token"]}
        new_info["gt_names"] = info["gt_names"]
        new_info["valid_flag"] = info["valid_flag"]
        nusc_infos.append(new_info)

    print(f"generate samples: {len(nusc_infos)}")
    data = {"infos": nusc_infos}
    if info_save_path is None:
        info_save_path = osp.join(out_dir, f"{info_prefix}_infos_train.pkl")

    with open(info_save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
