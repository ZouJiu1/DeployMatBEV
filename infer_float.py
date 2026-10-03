"""align bpu validation tools, Only support int-infer."""
import os
import cv2
import torch
import shutil
import argparse
import matplotlib
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import horizon_plugin_pytorch as horizon
from hat.registry import OBJECT_REGISTRY

from hat.registry import build_from_registry
from hat.utils.config import Config
from hat.utils.logger import init_logger
from hat.utils.checkpoint import load_state_dict

from ocamcamera import OcamCamera
from lmdbdata.fisheye_carla_dataset import *
from model.view_transformerFisheye import *
from model.fcos3d_goyu_metric import *
# from model.fisheye_lss import FisheyeLSSTransform
from model.fast_scnn import FastSCNNNeck_Fisheye
from model.multi_views import BevFeatureRotate_Fisheye
from model.target import CenterPointTarget_Fisheye

from hat.visualize.nuscenes import NuscenesViz
from hat.core.nus_box3d_utils import bbox_ego2bev, bbox_ego2img, bbox_to_corner

fileAbspath = __file__
fileName = os.path.basename(__file__)
abspath = fileAbspath.replace(fileName, "")

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

    def distance(self, c0, c1):
        return np.sqrt((c0[0] - c1[0])*(c0[0] - c1[0]) + (c0[1] - c1[1]) * (c0[1] - c1[1]))

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
            # if not self.checkCanDraw(corners_2d):
            #     continue
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

        cols = 2
        fig, axes = plt.subplots(
            int(np.ceil(n / cols)), cols, figsize=(16*2, 24*2)
        )
        axes = axes.flatten()
# [[13.557270050048828, -0.013530731201171875, -1.3188555240631104, 1.9289535284042358, 4.549752235412598, 1.5383068323135376, 0.021975034847855568, 0.00010173127520829439, 0.0006739980308339, 0.3186468183994293, 0.0], 
#  [-5.543117523193359, 4.077919006347656, -1.3188846111297607, 1.9383691549301147, 4.650774955749512, 1.5498931407928467, 3.069192886352539, -0.00033487225300632417, -4.5484361180569977e-05, 0.29078978300094604, 0.0], 
#  [-9.625579833984375, 0.007747650146484375, -1.3163961172103882, 1.9155628681182861, 4.5128703117370605, 1.5357016324996948, 0.009655393660068512, 6.100683094700798e-06, 0.00043967991950921714, 0.20638833940029144, 0.0], 
#  [9.587677001953125, 3.835723876953125, -1.2865312099456787, 2.0050930976867676, 4.731571197509766, 1.7828693389892578, 3.10671329498291, -0.00031733998912386596, -0.0006128593813627958, 0.1962326616048813, 0.0], 
#  [-40.790775299072266, 3.862751007080078, -1.3137496709823608, 1.9428598880767822, 4.630600929260254, 1.5369415283203125, -3.0812575817108154, -0.0001096886262530461, -0.0003747854789253324, 0.14526639878749847, 0.0], 
#  [-16.752239227294922, 3.9141311645507812, -1.3226901292800903, 1.9237579107284546, 4.544885158538818, 1.5343012809753418, 3.1273603439331055, -0.00013120780931785703, -9.60015386226587e-05, 0.1426653265953064, 0.0], 
#  [20.089839935302734, 4.045654296875, -1.3244787454605103, 1.9182798862457275, 4.546694278717041, 1.5365890264511108, 3.1196837425231934, -0.00020334879809524864, -0.00023715819406788796, 0.1064942255616188, 0.0], 
#  [9.560958862304688, 3.8275833129882812, -1.0256365537643433, 1.9935044050216675, 4.806850433349609, 2.11152982711792, 3.0398783683776855, -0.0017047458095476031, -0.0007326190243475139, 0.6522910594940186, 1.0], 
#  [-15.215164184570312, 0.1619110107421875, -0.8589515686035156, 2.0772347450256348, 5.983209609985352, 2.4412386417388916, 0.014527425169944763, -0.0015519424341619015, -0.0008818427450023592, 0.6133444905281067, 1.0], 
#  [-15.163040161132812, 0.2822761535644531, -1.0146116018295288, 2.4042131900787354, 6.7368316650390625, 2.201868772506714, 0.022685322910547256, -0.00022977007029112428, 2.737783688644413e-05, 0.1870032101869583, 2.0], 
#  [-17.58727264404297, 8.796905517578125, -1.0465924739837646, 0.37506771087646484, 0.375148206949234, 1.8606226444244385, 0.8158615827560425, 0.0003651438164524734, -0.00012684421380981803, 0.12851689755916595, 4.0], 
#  [28.029735565185547, -4.014701843261719, -1.032493233680725, 0.3750580847263336, 0.37513864040374756, 1.8606767654418945, -3.065910816192627, 0.00017382297664880753, -0.00015396845992654562, 0.12599408626556396, 4.0], 
#  [-5.605800628662109, 4.009601593017578, -1.2687853574752808, 0.8584752082824707, 1.891855239868164, 1.607896089553833, 3.071794033050537, -0.00048289899132214487, 0.00019173475448042154, 0.17014001309871674, 5.0], 
#  [-14.422466278076172, -0.026386260986328125, -1.3113209009170532, 0.8620058298110962, 1.832174301147461, 1.6067599058151245, -0.022206958383321762, -0.002321634441614151, 0.000136311020469293, 0.1598459929227829, 5.0]]
#         kk = [i.cpu().numpy().tolist() for i in bboxes]
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
        # default=r'/home/Desktop/fcos3d/infer_in/1533151669612404', 
        # default=r'/home/Desktop/fcos3d/infer_in/1693906258933460', 
        default=r'/home/Desktop/zoujiu/horizon/demo/fisheye3dod_demo/16', 
        # default = r'/home/Desktop/fisheyedod/horizon/demo/tmp/10',
        # default = r'/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ/10',
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
        # action="store_true",
        default=False,
        type=bool,
        help="Whether check the mlir model forward with example input.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        # default=r'/home/Desktop/fisheyedod/horizon/ckptsave/float-checkpoint-last.pth.tar',
        help="checkpoint path used to predict",
    )
    parser.add_argument(
        "--ocam_path",
        type=str,
        default=os.path.join(abspath, "./lmdbdata/calib_results.txt"),
        help="ocam_path",
    )
    parser.add_argument(
        "--learning_point_version",
        type = int,
        default='3',
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

    infer_float_cfg = config.get("infer_float_cfg")

    # get model inputs
    if args.model_inputs is not None:
        input_path = args.model_inputs
    else:
        input_path = infer_float_cfg.get("input_path")
        if args.use_dataset:
            gen_inputs_cfg = infer_float_cfg.get("gen_inputs_cfg")
            assert (
                gen_inputs_cfg is not None
            ), "You must set gen_inputs_cfg in infer_float_cfg when use dataset."
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

    prepare_inputs = infer_float_cfg.get("prepare_inputs", defualt_prepare_inputs)
    prepared_inputs = prepare_inputs(input_path)

    # build data transforms
    transforms = infer_float_cfg.get("transforms", None)

    if transforms is not None:
        transforms = build_from_registry(transforms)
        transforms = torchvision.transforms.Compose(transforms)

    # build model and load ckpt
    infer_float_cfg['model']['view_transformer']['ocam_path'] = str(args.ocam_path)
    if 'ocam_path' in infer_float_cfg['viz_func'].keys():
        infer_float_cfg['viz_func']['ocam_path'] = str(args.ocam_path)
    infer_float_cfg['model_convert_pipeline']['converters'][0]['checkpoint_path'] = args.ckpt
    infer_float_cfg['model']['view_transformer']['learning_point_version'] = str(args.learning_point_version)
    if int(args.learning_point_version) >= 9:
        infer_float_cfg['model']['view_transformer']['hidden_matmul'] = infer_float_cfg['model']['view_transformer']['grid_size'][0]
    else:
        infer_float_cfg['model']['view_transformer']['hidden_matmul'] = 256
    infer_float_cfg = build_from_registry(infer_float_cfg)
    model = infer_float_cfg['model']
    model = infer_float_cfg['model_convert_pipeline'].converters[0](model)
    # model = build_from_registry(config.get("model"))
    # load_state_dict(model, infer_cfg.get("modelPath"), check_hash=True)
    model.eval()

    viz_func = build_from_registry(infer_float_cfg.get("viz_func"))
    if isinstance(viz_func, list):
        for i, f in enumerate(viz_func):
            viz_func[i] = partial(f, save_path=args.save_path)
    else:
        viz_func = partial(viz_func, save_path=args.save_path)

    for i in os.listdir(args.save_path):
        pth = os.path.join(args.save_path, i)
        # if os.path.isdir(pth):
            # shutil.rmtree(pth)
        # if os.path.isfile(pth):
            # os.remove(pth)

    process_inputs = infer_float_cfg.get("process_inputs")
    process_outputs = infer_float_cfg.get("process_outputs")
    cam_names = [
            "fisheye_camera_front",
            "fisheye_camera_left",
            "fisheye_camera_rear",
            "fisheye_camera_right"
    ]
    # /root/.local/lib/python3.10/site-packages/hat/models/ir_modules/hbir_module.py
    for prepared_input in prepared_inputs:
        model_input_col, vis_inputs_col = process_inputs(prepared_input, transforms)

        # image = np.load(r'/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ/10/images.npy')
        # model_input_col['img'] = torch.from_numpy(image)
        # now = np.load(r'/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ/10/lidar2cam.npy')
        # model_input_col['lidar2cam'] = torch.from_numpy(now)
        # vis_inputs_col['meta']['lidar2cam'] = now
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
