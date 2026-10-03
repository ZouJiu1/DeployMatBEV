"""align bpu validation tools, Only support int-infer."""

import os
import torch
import shutil
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from hat.utils.config import Config
from hat.utils.logger import init_logger
from hat.registry import build_from_registry
from hat.core.nus_box3d_utils import bbox_bev2ego, bbox_ego2bev
from hat.core.cam_box3d import CameraInstance3DBoxes
from hat.visualize.cam3d import Cam3dViz, plot_rect3d_on_img

from lmdbdata.fisheye_carla_dataset import *
from model.view_transformerFisheye import *
from model.fcos3d_goyu_metric import *
# from model.fisheye_lss import FisheyeLSSTransform
from model.fast_scnn import FastSCNNNeck_Fisheye
from model.multi_views import BevFeatureRotate_Fisheye
from model.target import CenterPointTarget_Fisheye

from ocamcamera import OcamCamera
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion
import cv2

fileAbspath = __file__
fileName = os.path.basename(__file__)
abspath = fileAbspath.replace(fileName, "")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=False,
        default=os.path.join(abspath, "config/bev_lss_efficientnetb0_multitask_nuscenes.py"),
        help="train config file path",
    )
    parser.add_argument('--savePath', default=os.path.join(abspath, "delete"))
    return parser.parse_args()

class BoxVisualize(NuScenesBox):
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self,
                 center: list[float],
                 size: list[float],
                 orientation: Quaternion,
                 label=np.nan,
                 score=np.nan,
                 velocity=np.nan,
                 name=None,
                 token=None,
                 omni_ocam=None):
        super(BoxVisualize, self).__init__(center,
                            size,
                            orientation,
                            label,
                            score,
                            velocity,
                            name,
                            token)
        self.omni_ocam = omni_ocam

    def render_cv2(self,
                   im: np.ndarray,
                   corners: np.ndarray,
                   colors: tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
                   linewidth: int = 2) -> None:
        """
        Renders box using OpenCV2.
        :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
        :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
        :param linewidth: Linewidth for plot.
        """

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(im,
                         (int(prev[0]), int(prev[1])),
                         (int(corner[0]), int(corner[1])),
                         color, linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            cv2.line(im,
                     (int(corners.T[i][0]), int(corners.T[i][1])),
                     (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                     colors[2][::-1], linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0][::-1])
        draw_rect(corners.T[4:], colors[1][::-1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(im,
                 (int(center_bottom[0]), int(center_bottom[1])),
                 (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                 colors[0][::-1], linewidth)

def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d: Points in shape (N, 3)
        proj_mat: Transformation matrix between coordinates.
        with_depth: Whether to keep depth in the output.
            Defaults to False.

    Returns:
        Points in image coordinates,
            with shape [N, 2] if `with_depth=False`, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, (
        "The dimension of the projection"
        f" matrix should be 2 instead of {len(proj_mat.shape)}."
    )
    d1, d2 = proj_mat.shape[:2]
    assert (
        (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (d1 == 4 and d2 == 4)
    ), ("The shape of the projection matrix" f" ({d1}*{d2}) is not supported.")
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype
        )
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res

def bbox_to_corner(
    bboxes: torch.Tensor, score_thresh: float = 0.0
) -> tuple[list[np.array], list[float], list[float]]:
    """Get 3dbbox corner.

    Args:
        bboxes: Meta info for bbox. Shape as (n, 11) or (n, 10).
        score_thresh: Theshold for filtering bbox with low score.
    """

    decoded_bbox = []
    scores = []
    cat_ids = []
    for bbox in bboxes:
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()

        # score = bbox[9]
        # if score < score_thresh:
        #     continue
        w, l, h = bbox[3:6]
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        # Rotate
        yaw = bbox[6]
        rot = Quaternion(axis=[0, 0, 1], radians=yaw)

        corners = np.dot(rot.rotation_matrix, corners)

        # Translate
        x, y, z = bbox[:3]
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z
        corners = np.transpose(corners, (1, 0))
        decoded_bbox.append(corners)
        # if len(bbox) == 11:
        #     # scores.append(bbox[9])
        #     cat_ids.append(bbox[10])
        # else:
        #     # scores.append(1.0)
        #     cat_ids.append(bbox[9])

    return decoded_bbox, 0, 0

if __name__ == "__main__":
    args = parse_args()

    config = Config.fromfile(args.config)
    init_logger(f".hat_logs/{config.task_name}_infer_viz")

    # config._cfg_dict['data_loader']['dataset']['transforms'] = None
    bevSize = config._cfg_dict['bev_size']
    config._cfg_dict['data_loader']['dataset']['transforms'] = [
            # dict(type="MultiViewsImgResize", size=(396, 704)),
            # # dict(type="MultiViewsImgResize", size=(400, 400)),
            # dict(type="MultiViewsImgCrop", size=(256, 704)),
            # dict(type="MultiViewsImgFlip", prob=0.5),
            dict(
                type="MultiViewsImgTransformWrapper",
                transforms=[
                    dict(type="PILToTensor"),
                    # dict(type="BgrToYuv444", rgb_input=True),
                    # dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ]
    config._cfg_dict['data_loader']['batch_size'] = 1
    config._cfg_dict['num_workers'] = 1
    config._cfg_dict['shuffle'] = True

    data_loader = build_from_registry(config.get("data_loader"))
    camviz = Cam3dViz()

    for i in os.listdir(args.savePath):
        pth = os.path.join(args.savePath, i)
        if os.path.isdir(pth):
            shutil.rmtree(pth)
        else:
            os.remove(pth)
    cam_fov = 220
    ocam_path=os.path.join(abspath, "lmdbdata/calib_results.txt")
    omni_ocam = OcamCamera(filename=ocam_path, fov=cam_fov)
    use_box_not_camviz = 'box'
    out_dir = args.savePath

    length = len(data_loader)

    for idx, data in enumerate(data_loader):
        inputs =  {}
        result = [{}]

        # if idx % 100 != 0:
        #     continue
        for i, img in enumerate(data['img']):
            inputs['filename'] = "_".join(data['img_name'][0][i].split(os.sep)[-2:])
            inputs['img'] = data['img'][i].permute((1, 2, 0)).cpu().numpy()
            bev_bboxes_labels = bbox_bev2ego(data['bev_bboxes_labels'][0], bevSize)
            bev_bboxes_labels = torch.from_numpy(np.array(bev_bboxes_labels))

            bboxes = torch.from_numpy(data['gt_boxes_3d'][0])

            img = inputs['img'].copy()
            h, w, c = img.shape
            filename = inputs['filename']
            lidar2cam = data['lidar2cam'][i]
            # /home/Desktop/horizon/horizon_j6_open_explorer_v3.8.1-py310_20260326/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/infer_hbir.py
            # /usr/local/lib/python3.10/dist-packages/hat/visualize/nuscenes.py NuscenesViz
            if use_box_not_camviz == 'box':
                pred_img = img.copy()
                bboxes = bboxes.cpu().numpy()
                render = False
                for box in bboxes:
                    center = box[:3]
                    dim = np.array(box[3:6])   #[[1, 0, 2]].tolist() # width, length, width
                    # center[2] = center[2] + dim[2]*0.5
                    yaw = box[6]
                    xo = BoxVisualize(center=center,
                                        size=dim,
                                        orientation=Quaternion(axis=[0,0,1], angle=yaw),
                                        omni_ocam = omni_ocam,
                                    ) # need gravity center
                    corners = view_points(xo.corners(), lidar2cam, normalize=False)#[:2, :]

                    points_3d = np.transpose(corners)
                    mapx, mapy = omni_ocam.world2cam(points_3d.T)
                    corners = np.stack([mapx, mapy, points_3d[:, 2]], axis=-1).T
                    keep = (corners[0, :] >= 0) & \
                        (corners[0, :] < w) & \
                        (corners[1, :] >= 0) & \
                        (corners[1, :] < h)
                    cor = corners[:, keep]
                    if len(cor[0, :]) >= 5:
                        xo.render_cv2(pred_img, corners)
                        render = True

                if render and out_dir is not None:
                    print(out_dir, filename)
                    result_path = os.path.join(out_dir, filename)
                    os.makedirs(result_path, exist_ok=True)
                    img_pil = Image.fromarray(img)
                    img_pil.save(os.path.join(result_path, "img.png"))
                    pred_img_pil = Image.fromarray(pred_img)
                    pred_img_pil.save(os.path.join(result_path, "pred.png"))
                else:
                    plt.imshow(pred_img)
            elif use_box_not_camviz == "cam2viz":
                pred_img = img.copy()
                lidar2cam = lidar2cam.cpu().numpy()
                cambox = []
                for box in bev_bboxes_labels:
                    box = box.cpu().numpy()
                    nubox = NuScenesBox(center=box[:3], size=box[3:6], 
                                        orientation=Quaternion(axis=[0,0,1], angle=box[6]))
                    nubox.rotate(Quaternion._from_matrix(lidar2cam[:3, :3], atol=1e-4))
                    nubox.translate(lidar2cam[:3, 3])
                    # kk = nubox.orientation.yaw_pitch_roll
                    yaw = np.array([nubox.orientation.yaw_pitch_roll[0]])
                    lhw = nubox.wlh[[1, 2, 0]]
                    nubox.center[1] += lhw[1] * 0.5
                    cambox.append(np.concatenate([nubox.center, lhw, -yaw], axis = 0))
                cambox = np.array(cambox)

                # tmp = np.stack(bbox_ego2bev(bboxes, bevSize))
                # kk = data['bev_bboxes_labels'][0][:, :7] == tmp
                # ww = np.min(data['bev_bboxes_labels'][0][:, :7] - tmp)

                # /usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py
                # def get_cam_bboxes(): lhw = wlh[..., [1, 2, 0]]   def get_bev_bboxes()
                bboxes3d = CameraInstance3DBoxes(tensor=cambox, box_dim=7)    # need bottom center, not gravity center
                inputs['lidar2cam'] = [data['lidar2cam'][i]]
                result[0]['bboxes'] = bboxes3d
                result[0]['scores'] = torch.ones_like(bboxes3d.tensor)[:, 0]
                
                corners_3d = bboxes3d.corners
                num_bbox = corners_3d.shape[0]
                points_3d = corners_3d.reshape(-1, 3)
                if not isinstance(lidar2cam, torch.Tensor):
                    lidar2cam = torch.from_numpy(np.array(lidar2cam))

                assert lidar2cam.shape == torch.Size([3, 3]) or lidar2cam.shape == torch.Size(
                    [4, 4]
                )
                lidar2cam = lidar2cam.float().cpu()

                # project to 2d to get image coords (uv)
                # uv_origin = points_cam2img(points_3d, lidar2cam, with_depth=True)
                # uv_origin = view_points(points_3d.T, lidar2cam, normalize=False).T#[:2, :]
                # uv_origin = (uv_origin - 1).round()
                uv_origin = points_3d

                if isinstance(uv_origin, torch.Tensor):
                    uv_origin = uv_origin.cpu().numpy()
                mapx, mapy = omni_ocam.world2cam(uv_origin.T)
                uv_origin = np.stack([mapx, mapy, uv_origin[:, 2]], axis=-1)

                imgfov_pts_3d = uv_origin.reshape(num_bbox, 8, 3)

                plot = []
                for index, pts_3d in enumerate(imgfov_pts_3d):
                    keep = (pts_3d[:, 0] >= 0) &\
                        (pts_3d[:, 0] < w) &\
                        (pts_3d[:, 1] >= 0) &\
                        (pts_3d[:, 1] < h) &\
                        (pts_3d[:, 2] > 0)
                    pts_3d_ = pts_3d[keep, :]
                    if len(pts_3d_) >= 5:
                        plot.append(pts_3d)
                if len(plot) > 0:
                    plot = np.array(plot)[:, :, :2]

                    pred_img = plot_rect3d_on_img(pred_img, plot.shape[0], plot, (241, 101, 72), 1)

                    if out_dir is not None:
                        print(out_dir, filename)
                        result_path = os.path.join(out_dir, filename)
                        os.makedirs(result_path, exist_ok=True)
                        img_pil = Image.fromarray(img)
                        img_pil.save(os.path.join(result_path, "img.png"))
                        pred_img_pil = Image.fromarray(pred_img)
                        pred_img_pil.save(os.path.join(result_path, "pred.png"))
                    else:
                        plt.imshow(pred_img)
                # camviz(inputs, result, args.savePath, 0.1)
                # kk = 0
            else:
                pred_img = img.copy()
                lidar2cam = lidar2cam.cpu().numpy()
                bboxes = bboxes
                uv_origin, _, _ = bbox_to_corner(bboxes=bboxes, score_thresh=0.1)
                num_bbox = len(uv_origin)
                uv_origin = np.array(uv_origin)
                uv_origin = np.reshape(uv_origin, (-1, 3))

                uv_origin = view_points(uv_origin.T, lidar2cam, normalize=False).T
                if isinstance(uv_origin, torch.Tensor):
                    uv_origin = uv_origin.cpu().numpy()
                mapx, mapy = omni_ocam.world2cam(uv_origin.T)
                uv_origin = np.stack([mapx, mapy, uv_origin[:, 2]], axis=-1)

                imgfov_pts_3d = uv_origin.reshape(num_bbox, 8, 3)

                plot = []
                for index, pts_3d in enumerate(imgfov_pts_3d):
                    keep = (pts_3d[:, 0] >= 0) &\
                        (pts_3d[:, 0] < w) &\
                        (pts_3d[:, 1] >= 0) &\
                        (pts_3d[:, 1] < h) &\
                        (pts_3d[:, 2] > 0)
                    pts_3d_ = pts_3d[keep, :]
                    if len(pts_3d_) >= 5:
                        plot.append(pts_3d)
                if len(plot) > 0:
                    plot = np.array(plot)[:, :, :2]

                    pred_img = plot_rect3d_on_img(pred_img, plot.shape[0], plot, (241, 101, 72), 1)

                    if out_dir is not None:
                        print(out_dir, filename)
                        result_path = os.path.join(out_dir, filename)
                        os.makedirs(result_path, exist_ok=True)
                        img_pil = Image.fromarray(img)
                        img_pil.save(os.path.join(result_path, "img.png"))
                        pred_img_pil = Image.fromarray(pred_img)
                        pred_img_pil.save(os.path.join(result_path, "pred.png"))
                    else:
                        plt.imshow(pred_img)
                # camviz(inputs, result, args.savePath, 0.1)
                # kk = 0

        if idx == 2:
            break
        # break
