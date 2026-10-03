# Copyright (c) Horizon Robotics. All rights reserved.

import json
import logging
import os
from typing import Sequence

import numpy as np
import torch

from hat.core.nms.box3d_nms import nms_bev_multiclass
from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import (
    all_gather_object,
    get_dist_info,
    rank_zero_only,
)
import sys
fileAbspath = __file__
fileName = os.path.basename(__file__)
abspath = fileAbspath.replace(fileName, "")
predir = os.path.abspath(os.path.join(abspath, ".."))

sys.path.append(predir)
from hat.utils.package_helper import require_packages
from hat.metrics.metric import EvalMetric
from model.fisheye3dod_eval import Fisheye3DODEval

try:
    import pyquaternion
    from nuscenes.eval.common.utils import Quaternion
    # from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box as NuScenesBox
except ImportError:
    Quaternion = None
    config_factory = None
    NuScenesEval = None
    NuScenes = None
    NuScenesBox = None
    pyquaternion = None

from lmdbdata.classesLabel import CLASSES

logger = logging.getLogger(__name__)

__all__ = ["Fcos3dGOYUMetric", "Fcos3dGOYUMultiCamMetric"]

DefaultAttribute = {}
# DefaultAttribute = {
#     "car": "vehicle.parked",
#     "pedestrian": "pedestrian.moving",
#     "trailer": "vehicle.parked",
#     "truck": "vehicle.parked",
#     "bus": "vehicle.moving",
#     "motorcycle": "cycle.without_rider",
#     "construction_vehicle": "vehicle.parked",
#     "bicycle": "cycle.without_rider",
#     "barrier": "",
#     "traffic_cone": "",
# }
for C in CLASSES:
    DefaultAttribute[C] = C

@OBJECT_REGISTRY.register
class Fcos3dGOYUMetric(EvalMetric):
    """Evaluation Nuscenes Detection.

    Args:
        name: Name of this metric instance for display.
        data_root: Data path of nuscenes data.
        version: Version of nuscenes data.
                 Choosen from ['v1.0-mini', 'v1.0-tranval'].
        save_prefix: Path to save result.
        verbose: Wether output verbose log.
        eval_version: Eval version.
        use_lidar: Whether use lidar bbox.
        classes: List of class name.
        use_ddp: Whether use ddp to eval metric.
    """

    @require_packages(
        "nuscenes",
        "pyquaternion",
        raise_msg="Please `pip3 install nuscenes-devkit pyquaternion`",
    )
    def __init__(
        self,
        name: str = "NuscenesMetric",
        data_root: str = "",
        version: str = "v1.0-mini",
        save_prefix: str = "./WORKSPACE/results",
        verbose: bool = True,
        use_lidar: bool = False,
        classes: Sequence[str] = None,
        use_ddp: bool = True,
        trans_lidar_dim: bool = False,
        trans_lidar_rot: bool = True,
        meta_key="meta",
        lidar_key="lidar2ego",
    ):
        super(Fcos3dGOYUMetric, self).__init__(name)
        self.verbose = verbose
        self.save_prefix = save_prefix
        self.use_ddp = use_ddp
        self.meta_key = meta_key
        self.lidar_key = lidar_key

        self.trans_lidar_dim = trans_lidar_dim
        self.trans_lidar_rot = trans_lidar_rot

        if self.use_ddp:
            self.res_path = os.path.join(self.save_prefix, "results_nusc.json")
        else:
            global_rank, _ = get_dist_info()
            self.res_path = os.path.join(
                self.save_prefix, "results_nusc_{}.json".format(global_rank)
            )
        self.version = version
        self.data_root = data_root
        self.ret = ["NDS", 0.0]
        self.nusc_annos = {}
        self.use_lidar = use_lidar
        if classes is None:
            self.CLASSES = CLASSES
        else:
            self.CLASSES = classes

    def _output_to_nusc_box(self, bboxes):
        box_list = []
        for bbox in bboxes:
            bbox = bbox.cpu().numpy()
            center = bbox[:3]
            dims = bbox[3:6]
            yaw = bbox[6]
            quat = Quaternion(axis=[0, 0, 1], radians=yaw)
            velocity = (*bbox[7:9], 0.0)
            cat_id = bbox[10]
            score = bbox[9]
            box = NuScenesBox(
                center,
                dims,
                quat,
                label=cat_id,
                score=score,
                velocity=velocity,
            )
            box_list.append(box)
        return box_list

    def _ego_to_global(self, bbox, meta):
        ego2global_translation = meta["ego2global_translation"]
        ego2global_rotation = meta["ego2global_rotation"]
        bbox.rotate(Quaternion(ego2global_rotation))
        bbox.translate(ego2global_translation)
        return bbox

    def compute(self):
        pass

    def _output_to_nusc_box_lidar(self, detection, with_velocity=True):
        """Convert the output to the box class in the nuScenes.

        Args:
            detection: Detection results.

                - boxes_3d: Detection bbox.
                - scores_3d: Detection scores.
                - labels_3d: Predicted box labels.

        Returns:
            list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
        """
        if isinstance(detection, dict):
            box3d = detection["bboxes"].detach().cpu().numpy()
            scores = detection["scores"].detach().cpu().numpy()
            labels = detection["labels"].detach().cpu().numpy()

        else:
            box3d = detection.cpu().numpy()
        if self.trans_lidar_rot is True:
            box3d[:, 6] = -box3d[:, 6] - np.pi / 2

        box_list = []
        for i in range(len(box3d)):
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box3d[i, 6])
            if with_velocity:
                velocity = (*box3d[i, 7:9], 0.0)
            else:
                velocity = (0, 0, 0)
            if isinstance(detection, dict):
                label = labels[i]
                score = scores[i]
            else:
                label = box3d[i, 10]
                score = box3d[i, 9]

            box = NuScenesBox(
                box3d[i, :3],
                box3d[i, 3:6]
                if self.trans_lidar_dim is False
                else box3d[i, [4, 3, 5]],
                quat,
                label=label,
                score=score,
                velocity=velocity,
            )
            box_list.append(box)
        return box_list

    def _lidar_nusc_box_to_global(
        self,
        info,
        box,
        classes,
        eval_configs,
    ):
        """Convert the box from ego to global coordinate.

        Args:
            info (dict): Info for a specific sample data, including the
                calibration information.
            boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
            classes (list[str]): Mapped classes in the evaluation.
            eval_configs (object): Evaluation configuration object.

        Returns:
            list: List of standard NuScenesBoxes in the global
                coordinate.
        """

        # Move box to ego vehicle coord system
        lidar2ego_rotation = self.lidar_key + "_rotation"
        lidar2ego_translation = self.lidar_key + "_translation"
        box.rotate(pyquaternion.Quaternion(info[lidar2ego_rotation]))
        box.translate(np.array(info[lidar2ego_translation]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            return None
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))

        return box

    def update(
        self,
        meta,
        pred_bboxes,
    ):
        meta = meta[self.meta_key]
        for m, bboxes in zip(meta, pred_bboxes):
            annos = []
            if self.use_lidar:
                bboxes_list = self._output_to_nusc_box_lidar(bboxes)
            else:
                bboxes_list = self._output_to_nusc_box(bboxes)
            sample_token = m["sample_token"]
            for box in bboxes_list:
                if self.use_lidar:
                    box = self._lidar_nusc_box_to_global(
                        m, box, self.CLASSES, self.eval_detection_configs
                    )
                else:
                    box = self._ego_to_global(box, m)
                if box is None:
                    continue
                name = self.CLASSES[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = DefaultAttribute[name]

                nusc_anno = {
                    "sample_token": sample_token,
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr,
                }
                annos.append(nusc_anno)
            self.nusc_annos[sample_token] = annos

    def _gather(self):
        global_rank, global_world_size = get_dist_info()
        global_output = [None for _ in range(global_world_size)]
        all_gather_object(global_output, self.nusc_annos)
        return global_output

    def reset(self):
        self.nusc_annos = {}
        self.ret = ["NDS", 0.0]

    def get(self):
        logger.info(
            f"The length of self.nusc_annos is: {len(self.nusc_annos)}"
        )
        if len(self.nusc_annos) != 0:
            if self.use_ddp:
                nusc_annos = self._gather()
                self.nusc_annos = nusc_annos[0]
                for nusc_annos in nusc_annos[1:]:
                    self.nusc_annos.update(nusc_annos)
            self._get()
        return self.ret[0], self.ret[1]

    def _dump(self):
        modality = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
        nusc_sub = {
            "meta": modality,
            "results": self.nusc_annos,
        }
        logger.info(f"Results writes to {self.res_path}")
        if not os.path.exists(self.save_prefix):
            os.makedirs(self.save_prefix)

        with open(self.res_path, "w") as fs:
            json.dump(nusc_sub, fs)

    @rank_zero_only
    def _get(self):
        self._dump()
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=self.verbose
        )

        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=self.res_path,
            eval_set=eval_set_map[self.version],
            output_dir=self.save_prefix,
            verbose=self.verbose,
        )
        nusc_eval.main(render_curves=False)

        # record metrics
        metric_file = os.path.join(self.save_prefix, "metrics_summary.json")
        with open(metric_file) as fs:
            metrics = json.load(fs)

        log_info = ""
        log_info += "NDS: {:.4f}, mAP:{:.4f}\n".format(
            metrics["nd_score"], metrics["mean_ap"]
        )
        for name in self.CLASSES:
            log_info += "{}_AP:".format(name)
            for k, v in metrics["label_aps"][name].items():
                log_info += " [{}]:{:.4f} ".format(k, v)
            log_info += "\n"
        logger.info(log_info)
        self.ret[1] = metrics["nd_score"]


@OBJECT_REGISTRY.register
class Fcos3dGOYUMultiCamMetric(Fcos3dGOYUMetric):
    """Evaluation Nuscenes Detection for mono.

    Args:
        nms_threshold: NMS threshold for detection under same sample.
    """

    @require_packages(
        "nuscenes",
        "pyquaternion",
        raise_msg="Please `pip3 install nuscenes-devkit pyquaternion`",
    )
    def __init__(
        self, nms_threshold: float = 0.05, use_cpu: bool = False, \
        **kwargs
    ):
        super(Fcos3dGOYUMultiCamMetric, self).__init__(**kwargs)
        self.nms_threshold = nms_threshold
        self.use_cpu = use_cpu
        self.pred_result = {}
        self.gt_result = {}

    def _cam_to_ego(self, bbox, meta):
        cam2ego_translation = meta["sensor2ego_translation"]
        cam2ego_rotation = meta["sensor2ego_rotation"]
        # bbox.rotate(Quaternion(cam2ego_rotation))
        bbox.rotate(Quaternion._from_matrix(cam2ego_rotation, rtol=1e-04, atol=1e-04))
        bbox.translate(cam2ego_translation)
        return bbox

    def reset(self):
        self.nusc_annos = {}
        self.ret = ["NDS", 0.0]
        self.pred_result = {}
        self.gt_result = {}

    def _get_modality(self):
        modality = {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
        return modality

    def get_attr_name(self, attr_idx, label_name):
        """Get attribute from predicted index.

        This is a workaround to predict attribute when the predicted velocity
        is not reliable. We map the predicted attribute index to the one
        in the attribute set. If it is consistent with the category, we will
        keep it. Otherwise, we will use the default attribute.

        Args:
            attr_idx (int): Attribute index.
            label_name (str): Predicted category name.

        Returns:
            str: Predicted attribute name.
        """
        # TODO: Simplify the variable name
        AttrMapping_rev2 = [
            "cycle.with_rider",
            "cycle.without_rider",
            "pedestrian.moving",
            "pedestrian.standing",
            "pedestrian.sitting_lying_down",
            "vehicle.moving",
            "vehicle.parked",
            "vehicle.stopped",
            "None",
        ]
        if (
            label_name == "car"
            or label_name == "bus"
            or label_name == "truck"
            or label_name == "trailer"
            or label_name == "construction_vehicle"
        ):
            if (
                AttrMapping_rev2[attr_idx] == "vehicle.moving"
                or AttrMapping_rev2[attr_idx] == "vehicle.parked"
                or AttrMapping_rev2[attr_idx] == "vehicle.stopped"
            ):
                return AttrMapping_rev2[attr_idx]
            else:
                return DefaultAttribute[label_name]
        elif label_name == "pedestrian":
            if (
                AttrMapping_rev2[attr_idx] == "pedestrian.moving"
                or AttrMapping_rev2[attr_idx] == "pedestrian.standing"
                or AttrMapping_rev2[attr_idx]
                == "pedestrian.sitting_lying_down"
            ):
                return AttrMapping_rev2[attr_idx]
            else:
                return DefaultAttribute[label_name]
        elif label_name == "bicycle" or label_name == "motorcycle":
            if (
                AttrMapping_rev2[attr_idx] == "cycle.with_rider"
                or AttrMapping_rev2[attr_idx] == "cycle.without_rider"
            ):
                return AttrMapping_rev2[attr_idx]
            else:
                return DefaultAttribute[label_name]
        else:
            return DefaultAttribute[label_name]

    def _output_to_nusc_box(self, bboxes):
        box_list = []
        attrs = []
        for bbox in bboxes:
            bbox = bbox.cpu().numpy()
            center = bbox[:3]
            dims = bbox[3:6]
            yaw = bbox[6]
            quat = Quaternion(axis=[0, 0, 1], radians=yaw)
            # q2 = Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
            # quat = q2 * q1
            # center 3 dims 3 yaw 1 velocity 2 score 1 cat_id 1
            # if len(bbox) >= 10:
            velocity = ( bbox[7], bbox[8], 0.0 )
            # velocity = (*bbox[7:9], 0.0)
            # else:
            #     velocity = [0.0, 0.0]
            cat_id = bbox[-1]
            score = bbox[-2]
            # if len(bbox) == 12:
            #     attrs.append(bbox[11])
            # else:
            #     attrs.append(-1)
            box = NuScenesBox(
                center,
                dims,
                quat,
                label=cat_id,
                score=score,
                velocity=velocity,
            )
            box_list.append(box)
        return box_list, ""

    def update(
        self,
        metas,
        pred_bboxes,
    ):
        for i in range(len(pred_bboxes[1]['bev_det'])):
            token = metas['sample_token'][i]
            bboxes = torch.stack(pred_bboxes[1]['bev_det'][i])
            bboxes[:, 7:9] = 0.0
            if token not in self.pred_result.keys():
                self.pred_result[token] = [bboxes]
            else:
                if (bboxes[0, :6]==1.0).all():
                    continue
                tmp = torch.concat([self.pred_result[token][0], bboxes], dim = 0)
                self.pred_result[token] = [tmp]

        gt_bboxes = metas['gt_boxes_3d']
        for i in range(len(gt_bboxes)):
            token = metas['sample_token'][i]
            device = metas['img'].device
            center_dim_yaw_velocity = torch.from_numpy(metas['gt_boxes_3d'][i]).to(device=device)
            labels = torch.from_numpy(metas['gt_label'][i]).unsqueeze(-1).to(device=device)
            scores = torch.ones_like(labels)
            # if self.pred_result[0][0].shape[1] == 11:
            bboxes = torch.concat([center_dim_yaw_velocity, scores, labels], dim = 1)
            # else:
                # bboxes = torch.concat([center_dim_yaw, scores, labels], dim = 1)
            if token not in self.gt_result.keys():
                self.gt_result[token] = [bboxes]
            else:
                tmp = torch.concat([self.gt_result[token][0], bboxes], dim = 0)
                self.gt_result[token] = [tmp]

    def format_to_nusc_annos(self, results: dict):
        nusc_annos = {}
        for token, bboxes in results.items():
            annos = []
            box_list, _ = self._output_to_nusc_box(bboxes)
            for _, box in enumerate(box_list):
                # ========== 新增过滤逻辑 start ==========
                w, l, h = box.wlh
                # 过滤尺寸异常 或 置信度极低的预测框
                if w < 1e-3 or l < 1e-3 or h < 1e-3 or box.score < 1e-4:
                    continue
                # ========== 新增过滤逻辑 end ==========
                name = CLASSES[box.label]
                attribute_name = DefaultAttribute[name]
                nusc_anno = {
                    'sample_token': token,
                    'ego_translation': box.center.tolist(),  # ego pose defined as zero
                    'translation': box.center.tolist(),
                    'size': box.wlh.tolist(),
                    'rotation': box.orientation.elements.tolist(),
                    'velocity': box.velocity[:2].tolist(),
                    'detection_name': name,
                    'detection_score': box.score,
                    'attribute_name': attribute_name,
                }
                annos.append(nusc_anno)
            nusc_annos.update({token: annos})
        return nusc_annos

    def _merge_results_by_token(self, results: dict) -> dict:
        """Merge multi-camera results by sample token, avoid overwriting"""
        token2boxes = {}
        for bboxes_token in results:
            for token, box in bboxes_token.items():
                if token not in token2boxes.keys():
                    token2boxes[token] = []
                token2boxes[token].extend(box)
        merged = {}
        for token, box_list in token2boxes.items():
            box_list = [ i.to("cuda:0") for i in box_list ]
            merged[token] = torch.cat(box_list, dim=0)
        return merged

    def _gather_ddp(self):
        """Gather results from all ranks in DDP mode"""
        global_rank, global_world_size = get_dist_info()

        # Gather predictions
        gathered_pred = [None for _ in range(global_world_size)]
        all_gather_object(gathered_pred, self.pred_result)

        # Gather GT (deduplicate by token)
        gathered_gt = [None for _ in range(global_world_size)]
        all_gather_object(gathered_gt, self.gt_result)
        return gathered_pred, gathered_gt

    @rank_zero_only
    def __get(self, pred_token2boxes, gt_token2boxes):
        pred_annos = self.format_to_nusc_annos(pred_token2boxes)
        gt_annos = self.format_to_nusc_annos(gt_token2boxes)
        
        eval_title = '\n================== Fisheye3DOD Evaluation ==================\n'
        fisheye3d_eval = Fisheye3DODEval(pred_annos, gt_annos, ref_range=None)
        metrics_summary, result, details = fisheye3d_eval.main()
        result = eval_title + result
        logger.info(result)
        metric_dict = details
        # key = list(metric_dict.keys())
        # value = list(metric_dict.values())
        self.ret[1] = metric_dict["NDS"]
    
    def get(self):
        logger.info(
            f"Num samples: pred={len(self.pred_result)}, gt={len(self.gt_result)}"
        )
        if len(self.pred_result) > 0:
            # Gather DDP results
            pred_result, gt_result = self._gather_ddp()
            # Merge multi-camera results
            pred_token2boxes = self._merge_results_by_token(pred_result)
            gt_token2boxes = self._merge_results_by_token(gt_result)
            self.__get(pred_token2boxes, gt_token2boxes)
        return self.ret[0], self.ret[1]

    def fast_get(self):
        return self.ret[0], self.ret[1]

    def _nms(self):
        for token, annos in self.nusc_annos.items():
            bboxes = []
            scores = []
            labels = []
            if len(annos) == 0:
                continue
            for anno in annos:
                bbox = anno["bbox"]
                center = bbox.center.tolist()
                dims = bbox.wlh.tolist()
                rot = Quaternion(
                    bbox.orientation.elements.tolist()
                ).yaw_pitch_roll[0]
                nms_box = torch.tensor(
                    [center[0], center[1], dims[0], dims[1], rot],
                    dtype=torch.float32,
                )
                if not self.use_cpu:
                    nms_box = nms_box.cuda()
                bboxes.append(nms_box)
                scores.append(bbox.score)
                labels.append(bbox.label)
            bboxes = torch.stack(bboxes)
            scores = torch.tensor(scores)
            labels = torch.tensor(labels)
            if not self.use_cpu:
                scores = scores.cuda()
                labels = labels.cuda()
            keep = nms_bev_multiclass(
                bboxes, scores, labels, len(CLASSES), self.nms_threshold
            )
            new_annos = []
            for idx in keep:
                anno = annos[idx]
                box = anno["bbox"]
                name = CLASSES[box.label]
                attr = self.get_attr_name(int(anno["attribute"]), name)
                nusc_anno = {
                    "sample_token": anno["sample_token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr,
                }

                new_annos.append(nusc_anno)
            self.nusc_annos[token] = new_annos

    def _dump(self):
        self._nms()
        super(Fcos3dGOYUMultiCamMetric, self)._dump()
