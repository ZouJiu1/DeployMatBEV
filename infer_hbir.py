"""align bpu validation tools, Only support int-infer."""
import os
import argparse
from functools import partial
from hat.registry import OBJECT_REGISTRY
import horizon_plugin_pytorch as horizon
import torch
import torchvision
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from ocamcamera import OcamCamera
from hat.registry import build_from_registry
from hat.utils.config import Config
from hat.utils.logger import init_logger

from lmdbdata.fisheye_carla_dataset import *
from model.view_transformerFisheye import *
from model.fcos3d_goyu_metric import *
from model.fisheye_lss import FisheyeLSSTransform
from model.fast_scnn import FastSCNNNeck_Fisheye
from model.multi_views import BevFeatureRotate_Fisheye
from model.target import CenterPointTarget_Fisheye

from hat.visualize.nuscenes import NuscenesViz
from hat.core.nus_box3d_utils import bbox_ego2bev, bbox_ego2img, bbox_to_corner

@OBJECT_REGISTRY.register
class NuscenesViz_Fisheye(NuscenesViz):
    def __init__(self, ocam_path="", \
                cam_fov=220, **kwargs):
        super(NuscenesViz_Fisheye, self).__init__(**kwargs)
        self.DBL_MIN = 2.22507385850720138309023271733240406e-308
        self.eps = 1e-10
        self.thresh_STD = 0.26
        self.thresh_JOINPOINT = 0
        self.ocam_path = ocam_path
        self.ocam_fov = cam_fov
        self.omni_ocam = OcamCamera(filename=self.ocam_path, fov=self.ocam_fov)

    def _is_visible(corners_2d, corners_3d, imsize):
        visible = np.logical_and(
                    corners_2d[:, 0] > 0, corners_2d[:, 0] < imsize[1]
            )
        visible = np.logical_and(visible, corners_2d[:, 1] < imsize[0])
        visible = np.logical_and(visible, corners_2d[:, 1] > 0)
        visible = np.logical_and(visible, corners_3d[:, 2] > 1)
        return any(visible)

    def fisheyeOpencvProjectPoints(self, corners):
        if not isinstance(corners, np.ndarray):
            corners = np.array(corners)
        if corners.shape[0] == 3:
            corners = corners.T

        f = [self.intrinsic[0, 0], self.intrinsic[1, 1]];
        c = [self.intrinsic[0, 2], self.intrinsic[1, 2]];
        k1, k2, k3, k4 = self.D.ravel()
        pixels = []
        for i in range(len(corners)):
            coord = corners[i]
            if (abs(coord[2]) < self.DBL_MIN):
                coord[2] = 1
            # 1. 归一化
            xy = np.array([coord[0] / coord[2], coord[1] / coord[2]])
            # 2. 计算 theta
            r = np.sqrt(np.dot(xy, xy))
            theta = np.arctan(r)
            # 3. 畸变扭曲（Kannala-Brandt 核心）
            theta2 = np.power(theta, 2)
            theta4 = np.power(theta, 4)
            theta6 = np.power(theta, 6)
            theta8 = np.power(theta, 8)
            theta_d = theta * (1.0 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)
            # 4. 畸变后坐标
            inv_r = 1.0 / r if r > 1e-8 else 1
            cdist = theta_d * inv_r if r > 1e-8 else 1
            xyd1 = xy * cdist;
            # 5. 内参投影到像素
            x = xyd1[0] * f[0] + c[0]
            y = xyd1[1] * f[1] + c[1]
            pixels.append([x, y])
        return np.array(pixels)

    def distance(self, c0, c1):
        return np.sqrt((c0[0] - c1[0])*(c0[0] - c1[0]) + (c0[1] - c1[1]) * (c0[1] - c1[1]))

    def checkCanDraw(self, corners_coords):
        '''
        Check whether the line segment has intersections
        correct
        ---------
        |       |
        |       |
        |       |
        ---------
        wrong 
        -          -
        |  -    -  |
        |    -     |
        |  -   -   |
        -       -  |
        wrong
        ------------
        |          |
           |    |
             |
        |         |
        ------------
        '''
        corners = corners_coords.copy()

        # Check the sides
        sides = []
        for i in range(4):
            dis = self.distance(corners[:, i], corners[:, i + 4])
            sides.append(dis)
        sides = np.array(sides)
        norm = sides / self.width
        std_side = np.std(norm)

        # Check the fronts
        front0 = [self.distance(corners[:, 0], corners[:, 1]), self.distance(corners[:, 2], corners[:, 3])]
        front1 = [self.distance(corners[:, 0], corners[:, 3]), self.distance(corners[:, 1], corners[:, 2])]
        front0 = np.array(front0)
        front1 = np.array(front1)
        norm0 = front0 / self.width
        norm1 = front1 / self.width
        std_f0 = np.std( norm0 )
        std_f1 = np.std( norm1 )
        std_front = max(std_f0, std_f1)

        # Check the backs
        rear0 = [self.distance(corners[:, 4], corners[:, 5]), self.distance(corners[:, 6], corners[:, 7])]
        rear1 = [self.distance(corners[:, 4], corners[:, 7]), self.distance(corners[:, 5], corners[:, 6])]
        rear0 = np.array(rear0)
        rear1 = np.array(rear1)
        norm0 = rear0 / self.width
        norm1 = rear1 / self.width
        std_r0 = np.std( norm0 )
        std_r1 = np.std( norm1 )
        std_back = max(std_r0, std_r1)
        maxSTD = np.max([std_side, std_front, std_back])

        frontLeftTop  = [corners[0, 0], corners[1, 0]]
        frontRightTop = [corners[0, 1], corners[1, 1]]
        frontRightBottom = [corners[0, 2], corners[1, 2]]
        frontLeftBottom = [corners[0, 3], corners[1, 3]]
        rearLeftTop = [corners[0, 4], corners[1, 4]]
        rearRightTop = [corners[0, 5], corners[1, 5]]
        rearRightBottom = [corners[0, 6], corners[1, 6]]
        rearLeftBottom = [corners[0, 7], corners[1, 7]]

        # check the front up down back up down not join at a point
        k0 = (frontLeftTop[1] - frontRightTop[1]) / ((frontLeftTop[0] - frontRightTop[0]) + self.eps)
        k1 = (frontLeftBottom[1] - frontRightBottom[1]) / ((frontLeftBottom[0] - frontRightBottom[0]) + self.eps)
        k2 = (rearLeftTop[1] - rearRightTop[1]) / ((rearLeftTop[0] - rearRightTop[0]) + self.eps)
        k3 = (rearLeftBottom[1] - rearRightBottom[1]) / ((rearLeftBottom[0] - rearRightBottom[0]) + self.eps)

        b0 = frontLeftTop[1] - k0 * frontLeftTop[0]
        b1 = frontLeftBottom[1] - k1 * frontLeftBottom[0]
        b2 = rearLeftTop[1] - k2 * rearLeftTop[0]
        b3 = rearLeftBottom[1] - k3 * rearLeftBottom[0]

        x01_join = (b1 - b0) / (self.eps + k0 - k1)
        x02_join = (b2 - b0) / (self.eps + k0 - k2)
        x03_join = (b3 - b0) / (self.eps + k0 - k3)
        x12_join = (b2 - b1) / (self.eps + k1 - k2)
        x13_join = (b3 - b1) / (self.eps + k1 - k3)
        x23_join = (b3 - b2) / (self.eps + k2 - k3)

        join01 = ( x01_join > max(min(frontLeftTop[0], frontRightTop[0]), min(frontLeftBottom[0], frontRightBottom[0])) ) and \
                ( x01_join < min(max(frontLeftTop[0], frontRightTop[0]), max(frontLeftBottom[0], frontRightBottom[0])) )
        
        join02 = ( x02_join > max(min(frontLeftTop[0], frontRightTop[0]), min(rearLeftTop[0], rearRightTop[0])) ) and \
                ( x02_join < min(max(frontLeftTop[0], frontRightTop[0]), max(rearLeftTop[0], rearRightTop[0])) )
        
        join03 = ( x03_join > max(min(frontLeftTop[0], frontRightTop[0]), min(rearLeftBottom[0], rearRightBottom[0])) ) and \
                ( x03_join < min(max(frontLeftTop[0], frontRightTop[0]), max(rearLeftBottom[0], rearRightBottom[0])) )
        
        join12 = ( x12_join > max(min(frontLeftBottom[0], frontRightBottom[0]), min(rearLeftTop[0], rearRightTop[0])) ) and \
                ( x12_join < min(max(frontLeftBottom[0], frontRightBottom[0]), max(rearLeftTop[0], rearRightTop[0])) )
        
        join13 = ( x13_join > max(min(frontLeftBottom[0], frontRightBottom[0]), min(rearLeftBottom[0], rearRightBottom[0])) ) and \
                ( x13_join < min(max(frontLeftBottom[0], frontRightBottom[0]), max(rearLeftBottom[0], rearRightBottom[0])) )
        
        join23 = ( x23_join > max(min(rearLeftTop[0], rearRightTop[0]), min(rearLeftBottom[0], rearRightBottom[0])) ) and \
                ( x23_join < min(max(rearLeftTop[0], rearRightTop[0]), max(rearLeftBottom[0], rearRightBottom[0])) )

        retUD = np.sum( [ join01, join02, join03, join12, join13, join23 ] )


        # check the front left right back left right not join at a point
        k0 = (frontLeftTop[1] - frontLeftBottom[1]) / ((frontLeftTop[0] - frontLeftBottom[0]) + self.eps)
        k1 = (frontRightTop[1] - frontRightBottom[1]) / ((frontRightTop[0] - frontRightBottom[0]) + self.eps)
        k2 = (rearLeftTop[1] - rearLeftBottom[1]) / ((rearLeftTop[0] - rearLeftBottom[0]) + self.eps)
        k3 = (rearRightTop[1] - rearRightBottom[1]) / ((rearRightTop[0] - rearRightBottom[0]) + self.eps)

        b0 = frontLeftTop[1] - k0 * frontLeftTop[0]
        b1 = frontRightTop[1] - k1 * frontRightTop[0]
        b2 = rearLeftTop[1] - k2 * rearLeftTop[0]
        b3 = rearRightTop[1] - k3 * rearRightTop[0]

        x01_join = (b1 - b0) / (self.eps + k0 - k1)
        x02_join = (b2 - b0) / (self.eps + k0 - k2)
        x03_join = (b3 - b0) / (self.eps + k0 - k3)
        x12_join = (b2 - b1) / (self.eps + k1 - k2)
        x13_join = (b3 - b1) / (self.eps + k1 - k3)
        x23_join = (b3 - b2) / (self.eps + k2 - k3)

        join01 = ( x01_join > max(min(frontLeftTop[0], frontLeftBottom[0]), min(frontRightTop[0], frontRightBottom[0])) ) and \
                ( x01_join < min(max(frontLeftTop[0], frontLeftBottom[0]), max(frontRightTop[0], frontRightBottom[0])) )
        
        join02 = ( x02_join > max(min(frontLeftTop[0], frontLeftBottom[0]), min(rearLeftTop[0], rearLeftBottom[0])) ) and \
                ( x02_join < min(max(frontLeftTop[0], frontLeftBottom[0]), max(rearLeftTop[0], rearLeftBottom[0])) )
        
        join03 = ( x03_join > max(min(frontLeftTop[0], frontLeftBottom[0]), min(rearRightTop[0], rearRightBottom[0])) ) and \
                ( x03_join < min(max(frontLeftTop[0], frontLeftBottom[0]), max(rearRightTop[0], rearRightBottom[0])) )
        
        join12 = ( x12_join > max(min(frontRightTop[0], frontRightBottom[0]), min(rearLeftTop[0], rearLeftBottom[0])) ) and \
                ( x12_join < min(max(frontRightTop[0], frontRightBottom[0]), max(rearLeftTop[0], rearLeftBottom[0])) )
        
        join13 = ( x13_join > max(min(frontRightTop[0], frontRightBottom[0]), min(rearRightTop[0], rearRightBottom[0])) ) and \
                ( x13_join < min(max(frontRightTop[0], frontRightBottom[0]), max(rearRightTop[0], rearRightBottom[0])) )
        
        join23 = ( x23_join > max(min(rearLeftTop[0], rearLeftBottom[0]), min(rearRightTop[0], rearRightBottom[0])) ) and \
                ( x23_join < min(max(rearLeftTop[0], rearLeftBottom[0]), max(rearRightTop[0], rearRightBottom[0])) )

        retLR = np.sum( [ join01, join02, join03, join12, join13, join23 ] )

        # check the sides not join at a point
        k0 = (frontLeftTop[1] - rearLeftTop[1]) / ((frontLeftTop[0] - rearLeftTop[0]) + self.eps)
        k1 = (frontRightTop[1] - rearRightTop[1]) / ((frontRightTop[0] - rearRightTop[0]) + self.eps)
        k2 = (frontRightBottom[1] - rearRightBottom[1]) / ((frontRightBottom[0] - rearRightBottom[0]) + self.eps)
        k3 = (frontLeftBottom[1] - rearLeftBottom[1]) / ((frontLeftBottom[0] - rearLeftBottom[0]) + self.eps)

        b0 = frontLeftTop[1] - k0 * frontLeftTop[0]
        b1 = frontRightTop[1] - k1 * frontRightTop[0]
        b2 = frontRightBottom[1] - k2 * frontRightBottom[0]
        b3 = frontLeftBottom[1] - k3 * frontLeftBottom[0]

        x01_join = (b1 - b0) / (self.eps + k0 - k1)
        x02_join = (b2 - b0) / (self.eps + k0 - k2)
        x03_join = (b3 - b0) / (self.eps + k0 - k3)
        x12_join = (b2 - b1) / (self.eps + k1 - k2)
        x13_join = (b3 - b1) / (self.eps + k1 - k3)
        x23_join = (b3 - b2) / (self.eps + k2 - k3)

        join01 = ( x01_join > max(min(frontLeftTop[0], rearLeftTop[0]), min(frontRightTop[0], rearRightTop[0])) ) and \
                ( x01_join < min(max(frontLeftTop[0], rearLeftTop[0]), max(frontRightTop[0], rearRightTop[0])) )
        
        join02 = ( x02_join > max(min(frontLeftTop[0], rearLeftTop[0]), min(frontRightBottom[0], rearRightBottom[0])) ) and \
                ( x02_join < min(max(frontLeftTop[0], rearLeftTop[0]), max(frontRightBottom[0], rearRightBottom[0])) )
        
        join03 = ( x03_join > max(min(frontLeftTop[0], rearLeftTop[0]), min(frontLeftBottom[0], rearLeftBottom[0])) ) and \
                ( x03_join < min(max(frontLeftTop[0], rearLeftTop[0]), max(frontLeftBottom[0], rearLeftBottom[0])) )
        
        join12 = ( x12_join > max(min(frontRightTop[0], rearRightTop[0]), min(frontRightBottom[0], rearRightBottom[0])) ) and \
                ( x12_join < min(max(frontRightTop[0], rearRightTop[0]), max(frontRightBottom[0], rearRightBottom[0])) )
        
        join13 = ( x13_join > max(min(frontRightTop[0], rearRightTop[0]), min(frontLeftBottom[0], rearLeftBottom[0])) ) and \
                ( x13_join < min(max(frontRightTop[0], rearRightTop[0]), max(frontLeftBottom[0], rearLeftBottom[0])) )
        
        join23 = ( x23_join > max(min(frontRightBottom[0], rearRightBottom[0]), min(frontLeftBottom[0], rearLeftBottom[0])) ) and \
                ( x23_join < min(max(frontRightBottom[0], rearRightBottom[0]), max(frontLeftBottom[0], rearLeftBottom[0])) )

        ret = np.sum( [ join01, join02, join03, join12, join13, join23 ] )


        candraw = ( (( ret + retUD + retLR ) <= 0) & (maxSTD < self.thresh_STD) )

        return candraw

    def bbox_lidar2img(self, bboxes, lidar2cam, img_size, score_thresh):
        # img_size: (width, height)
        decoded_corners_3d, _, _ = bbox_to_corner(bboxes, score_thresh)
        img_bboxes = []
        self.width = img_size[0]
        self.height = img_size[1]
        for corners_3d in decoded_corners_3d:
            nbr_points = corners_3d.shape[0]
            corners_3d = np.transpose(corners_3d, (1, 0))
            corners_3d = np.concatenate((corners_3d, np.ones((1, nbr_points))))
            corners_3d = np.dot(lidar2cam, corners_3d)
            corners_2d = corners_3d[:3, :]

            mapx, mapy = self.omni_ocam.world2cam(corners_2d)
            corners_2d = np.stack([mapx, mapy])

            # corners_2d = np.transpose(corners_2d[:2, :], (1, 0))
            if -1 in corners_2d[0, :] or -1 in corners_2d[1, :]:
                continue
            if not self.checkCanDraw(corners_2d):
                continue
            corners_2d = corners_2d.T
            corners_2d[..., 0] = np.clip(corners_2d[..., 0], 0, self.width - 1)
            corners_2d[..., 1] = np.clip(corners_2d[..., 1], 0, self.height - 1)
            img_bboxes.append(corners_2d)
        return img_bboxes

    def draw_cam_img(self, axes: matplotlib.axes.Axes, imgs, homos, bboxes):
        for i, img in enumerate(imgs):
            name = img["name"]
            image = img["img"]
            ax = axes[i]
            ax.set_title(name)
            ax.grid(False)
            image = image.numpy().squeeze()
            image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
            h, w, c = image.shape
            img_bbox = self.bbox_lidar2img(
                bboxes, homos[i], (w, h), self.score_thresh
            )
            # img_bbox = img_bbox * 2
            # image = cv2.resize(image, (w*2, h*2))
            self.draw_bboxes(ax, img_bbox, linewidth=1)
            ax.imshow(image)
        return len(imgs)

    def __call__(self, imgs, preds, meta, save_path=None):
        num_imgs = len(imgs)
        n = num_imgs

        if "bev_det" in preds:
            bboxes = preds["bev_det"][0]
        elif "ego_det" in preds:
            bboxes = preds["ego_det"][0]
        elif "lidar_det" in preds:
            bboxes = preds["lidar_det"][0]
        else:
            bboxes = []

        if "bev_seg" in preds:
            mask = preds["bev_seg"][0]
            if self.use_bce:
                mask += 1
            n += 1
        else:
            mask = None

        if "ego2img" in meta:
            homos = meta["ego2img"]
        elif 'lidar2cam' in meta:
            homos = meta['lidar2cam']
        else:
            homos = meta["lidar2img"]
        if len(bboxes) != 0:
            n += 1

        cols = 3
        fig, axes = plt.subplots(
            int(np.ceil(n / cols)), cols, figsize=(16*2, 24*2)
        )
        axes = axes.flatten()
        now = self.draw_cam_img(axes[:6], imgs, homos, bboxes)

        cur_index = now
        if len(bboxes) != 0:
            if "bev_det" in preds:
                bev_bboxes = bbox_ego2bev(bboxes, self.bev_size)
                bev_bboxes, _, _ = bbox_to_corner(bev_bboxes)
                self.draw_bev_bboxes(axes[cur_index], bev_bboxes)

            else:
                ego_bboxes, _, _ = bbox_to_corner(bboxes.cpu().numpy())
                self.draw_ego_bboxes(axes[cur_index], ego_bboxes)
            cur_index += 1
        if mask is not None:
            self.draw_mask(axes[cur_index], mask)
            cur_index += 1

        self.sample_idx = 1
        if self.is_plot:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                result_path = os.path.join(
                    save_path, f"nusc_pred_{self.sample_idx}.jpg"
                )
                plt.savefig(result_path)
            else:
                plt.show()
        self.sample_idx += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=False,
        default=r'/home/Desktop/fisheyedod/horizon/config/bev_lss_efficientnetb0_multitask_nuscenes.py',
        help="train config file path",
    )
    parser.add_argument(
        "--model-inputs", type=str, 
        # default=r'/home/Desktop/fisheyedod/horizon/demo/tmp/10', 
        default=r'/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ/10',
        help="model input"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=r'/home/Desktop/fisheyedod/horizon/infer_out',
        help="save path for visualize output.",
    )
    parser.add_argument(
        "--use-dataset",
        action="store_true",
        help="Whether check the mlir model forward with example input.",
    )
    return parser.parse_args()

def defualt_prepare_inputs(infer_inputs):

    return [infer_inputs]

def _analyze_inputs(inputs):

    inputs_list = inputs.split(",")
    model_inputs = {}
    for each_input in inputs_list:
        name, values = each_input.split(":")
        model_inputs[name.strip()] = values.strip()

    return model_inputs

if __name__ == "__main__":
    args = parse_args()

    config = Config.fromfile(args.config)
    init_logger(f".hat_logs/{config.task_name}_infer_viz")
    horizon.march.set_march(config.get("march"))

    infer_cfg = config.get("infer_cfg")

    # get model inputs
    if args.model_inputs is not None:
        input_path = args.model_inputs
    else:
        input_path = infer_cfg.get("input_path")
        if args.use_dataset:
            gen_inputs_cfg = infer_cfg.get("gen_inputs_cfg")
            assert (
                gen_inputs_cfg is not None
            ), "You must set gen_inputs_cfg in infer_cfg when use dataset."
            dataset = build_from_registry(gen_inputs_cfg["dataset"])
            sample_idx = gen_inputs_cfg["sample_idx"]
            if len(sample_idx) == 1:
                sample_data = dataset[sample_idx[0]]
            else:
                sample_data = []
                for idx_ in range(sample_idx[0], sample_idx[1]):
                    sample_data.append(dataset[idx_])
            inputs_save_func = gen_inputs_cfg.get("inputs_save_func")
            inputs_save_func(sample_data, input_path)

    prepare_inputs = infer_cfg.get("prepare_inputs", defualt_prepare_inputs)
    prepared_inputs = prepare_inputs(input_path)

    # build data transforms
    transforms = infer_cfg.get("transforms", None)

    if transforms is not None:
        transforms = build_from_registry(transforms)
        transforms = torchvision.transforms.Compose(transforms)

    # build model and load ckpt
    model = build_from_registry(infer_cfg.get("model"))
    model.eval()

    viz_func = build_from_registry(infer_cfg.get("viz_func"))
    if isinstance(viz_func, list):
        for i, f in enumerate(viz_func):
            viz_func[i] = partial(f, save_path=args.save_path)
    else:
        viz_func = partial(viz_func, save_path=args.save_path)

    process_inputs = infer_cfg.get("process_inputs")
    process_outputs = infer_cfg.get("process_outputs")

    # /root/.local/lib/python3.10/site-packages/hat/models/ir_modules/hbir_module.py
    for prepared_input in prepared_inputs:
        model_input_col, vis_inputs_col = process_inputs(prepared_input, transforms)

        image = np.load(r'/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ/10/images.npy')
        model_input_col['img'] = torch.from_numpy(image)
        now = np.load(r'/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ/10/lidar2cam.npy')
        model_input_col['lidar2cam'] = torch.from_numpy(now)
        vis_inputs_col['meta']['lidar2cam'] = now

        with torch.no_grad():
            # output = r'/home/Desktop/PTQ2QAT/visinsex/fcos3d/data/1533151669612404/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151669612404.npy'
            # img = model_input['img'].detach().cpu().numpy()
            # img = np.asarray(img, dtype=np.float32)
            # np.save(output, img)
            # bin = np.clip(np.round(img / 0.007835294120013714), -128, 127).astype(np.int8)
            # bin.tofile("/home/Desktop/PTQ2QAT/visinsex/fcos3d/board_infer/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151669612404.bin")
            model_outputs = model(model_input_col)

        outputs = process_outputs(model_outputs, viz_func, vis_inputs_col)
        if outputs is not None:
            print(outputs)