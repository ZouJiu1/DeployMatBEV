###########################
# matmul version

import copy
import math
import os
import shutil

import cv2
import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import pil_to_tensor

try:
    from torchvision.transforms.functional_tensor import resize
except ImportError:
    # torchvision 0.18
    from torchvision.transforms._functional_tensor import resize

from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)
import sys

fileAbspath = __file__
fileName = os.path.basename(__file__)
abspath = fileAbspath.replace(fileName, "")

predir = os.path.abspath(os.path.join(abspath, ".."))

sys.path.append(predir)
import hat.data.datasets.nuscenes_dataset as NuscenesDataset
from lmdbdata.nusc_collatesFisheye import collate_nuscenesFisheye
from hat.metrics.mean_iou import MeanIOU
from hat.utils.aibenchmark_bpu.bev_process_bpu import (
    preprocess_data,
    process_bev_data,
    process_bev_metric,
)
from hat.utils.config import ConfigVersion
from lmdbdata.classesLabel import CLASSES
from lmdbdata.fisheye_carla_dataset import *
from model.hbir_module import HbirModuleFisheye
from model.fast_scnn import *
from model.multi_views import BevFeatureRotate_Fisheye
from model.target import CenterPointTarget_Fisheye

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "bev_lss_efficientnetb0_multitask_nuscenes"
batch_size_per_gpu = 1
device_ids = [0]
dataloader_workers = 0# batch_size_per_gpu  # per gpu
# ckpt_dir = "/home/Desktop/tmp_models/%s" % task_name
# ckpt_dir = os.path.join(predir, "model", "ckpt")
ckpt_dir = os.path.join(predir, "model", "ckpt11")
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_E
convert_mode = "jit-strip"
enable_amp = False

orig_shape = (3, 800, 800)

# resize_shape = (3, 396, 704)
# data_shape = (3, 256, 704)
# val_data_shape = (3, 256, 704)

resize_shape = (3, 800, 800)
data_shape = (3, 800, 800)
val_data_shape = (3, 800, 800)

feat_scale = ( 1 / 16.0, 1 / 16.0 )
feat_hw = (int(data_shape[1] * feat_scale[0]), int(data_shape[2] * feat_scale[1]))
vt_input_hw = (
    16,
    44,
)  # view transformer input shape for generationg reference points.

bn_kwargs = dict(eps=2e-5, momentum=0.1)

weight_decay = 0.01
start_lr = 2e-4
train_epochs = 36

num_views = 4
bev_size = (51.2, 51.2, 0.8)
grid_size = (128, 128)
dbound = [0.8, 52.0, 0.8]
# bev_size = (51.2, 51.2, 0.4)
# grid_size = (256, 256)
map_size = (15, 30, 0.15)
#dbound = [1.0, 61.0, 1.0]
homo_key="lidar2cam"

# TODOING
original_bevpool = False
original_gridsample = False
useLidar2cam = True
useSpherical = True
learning_point = True
num_points = 1
svd_rank = 16
num_tile = (2*2)
efficientnet_TYPE = "b0"

# task_map_size = (15, 30, 0.15)
qat_lr = 2e-5
qat_train_epochs = 5

data_rootdir = "/home/data/Fisheye3DODdataset/lmdb"
# data_rootdir = "/media/datasets/Fisheye3DODdataset/lmdb"
meta_rootdir = "do_not_need"

seg_classes_name = ["others", "divider", "ped_crossing", "Boundary"]
use_bce = False

if use_bce:
    seg_classes = 3
else:
    seg_classes = 3 + 1

depth = 64
feat_channels = 64
hidden_matmul = 256
learning_point_version = "11" # 1 2 3 9 10 11
if int(learning_point_version) >= 9:
    hidden_matmul = grid_size[0]
if learning_point:
    assert depth == feat_channels

cam_fov = 220
ocam_path = '/home/data/Fisheye3DODdataset/calib_results.txt'
ocam_path = '/media/datasets/Fisheye3DODdataset/calib_results.txt'

def get_grid_quant_scale(grid_shape, view_shape):
    max_coord = max(*grid_shape, *view_shape)
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = max(min(coord_shift, 8), 0)
    grid_quant_scale = 1.0 / (1 << coord_shift)
    return grid_quant_scale


view_shape = [data_shape[1] / 16, data_shape[2] / 16]
featview_shape = [view_shape[0] * 6, view_shape[1]]
grid_quant_scale = get_grid_quant_scale(grid_size, featview_shape)

depthview_shape = [6 * depth, view_shape[0] * view_shape[1]]
depth_quant_scale = get_grid_quant_scale(grid_size, depthview_shape)

# map_shape = [
#     int(task_map_size[1] * 2 / task_map_size[2]),
#     int(task_map_size[0] * 2 / task_map_size[2]),
# ]
# map_grid_quant_scale = get_grid_quant_scale(map_shape, view_shape)


# tasks = [
#     dict(name="car", num_class=1, class_names=["car"]),
#     dict(
#         name="truck",
#         num_class=2,
#         class_names=["truck", "construction_vehicle"],
#     ),
#     dict(name="bus", num_class=2, class_names=["bus", "trailer"]),
#     dict(name="barrier", num_class=1, class_names=["barrier"]),
#     dict(name="bicycle", num_class=2, class_names=["motorcycle", "bicycle"]),
#     dict(
#         name="pedestrian",
#         num_class=2,
#         class_names=["pedestrian", "traffic_cone"],
#     ),
# ]
# task_map_size = (15, 30, 0.15)
# map_shape = [
#     int(task_map_size[1] * 2 / task_map_size[2]),
#     int(task_map_size[0] * 2 / task_map_size[2]),
# ]
# map_grid_quant_scale = get_grid_quant_scale(map_shape, view_shape)
tasks = []
for i in CLASSES:
    tasks.append({'name':i, 'num_class':1, 'class_names':[i]})


if original_bevpool:
    view_transform=dict(
        type='FisheyeLSSTransform',
        in_channels=64,
        out_channels=64,
        image_size=[-1, -1],
        # feature_size=[25, 100],
        feature_size=[int(bev_size[0]*2/bev_size[2]), int(bev_size[1]*2/bev_size[2])],
        azimuth_range=[-math.radians(cam_fov/2), math.radians(cam_fov/2)],
        elevation_range=[-math.pi/4, math.pi/4],
        xbound=[-bev_size[0], bev_size[0], bev_size[2]],
        ybound=[-bev_size[1], bev_size[1], bev_size[2]],
        zbound=[-5.0, 5.0, 10.0],
        dbound=dbound,
        downsample=1,
        ocam_fov=cam_fov,
        ocam_path=ocam_path,
        num_views=num_views,
        )
else:
    view_transform=dict(
        type="LSSTransformerFisheye",
        in_channels=64,
        feat_channels=feat_channels,
        # z_range=(-10.0, 10.0),
        z_range=(-32.0, 32.0),
        depth=depth,
        num_points=num_points,
        bev_size=bev_size,
        grid_size=grid_size,
        num_views=num_views,
        grid_quant_scale=grid_quant_scale,
        depth_grid_quant_scale=depth_quant_scale,
        homo_key=homo_key,
        ocam_fov=cam_fov,
        ocam_path=ocam_path,
        azimuth_range=[-math.radians(cam_fov/2), math.radians(cam_fov/2)],
        elevation_range=[-math.pi/4, math.pi/4],
        dbound = dbound,
        useLidar2cam = useLidar2cam,
        useSpherical = useSpherical,
        feat_hw = feat_hw,
        original_gridsample = original_gridsample,
        feat_scale = feat_scale,
        svd_rank = svd_rank,
        num_tile = num_tile,
        learning_point = learning_point,
        learning_point_version = learning_point_version,
        hidden_matmul = hidden_matmul,
    )

model = dict(
    type="ViewFusion",
    bev_feat_index=-1,
    bev_upscale=2,
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type=efficientnet_TYPE,
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    neck=dict(
        type="FastSCNNNeck_Fisheye",
        in_channels=[112, 320],
        feat_channels=[64, 64],
        indexes=[-2, -1],
        bn_kwargs=bn_kwargs,
        scale_factor=2,
    ),
    view_transformer=view_transform,
    bev_transforms=[
        dict(
            type="BevFeatureRotate_Fisheye",
            bev_size=bev_size,
            rot=(-0.3925, 0.3925),
        ),
    ],
    bev_encoder=dict(
        type="BevEncoder",
        backbone=dict(
            type="efficientnet",
            bn_kwargs=bn_kwargs,
            model_type=efficientnet_TYPE,
            num_classes=1000,
            include_top=False,
            activation="relu",
            use_se_block=False,
            input_channels=64,
            quant_input=False,
        ),
        neck=dict(
            type="BiFPN",
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[2, 4, 8, 16, 32],
            stride2channels=dict({2: 16, 4: 24, 8: 40, 16: 112, 32: 320}),
            out_channels=48,
            num_outs=5,
            stack=3,
            start_level=0,
            end_level=-1,
            fpn_name="bifpn_sum",
            upsample_type="function",
            use_fx=True,
        ),
    ),
    bev_decoders=[
        # dict(
        #     type="BevSegDecoder",
        #     name="bev_seg",
        #     use_bce=use_bce,
        #     bev_size=bev_size,
        #     task_size=task_map_size,
        #     grid_quant_scale=map_grid_quant_scale,
        #     task_weight=10.0,
        #     head=dict(
        #         type="DepthwiseSeparableFCNHead",
        #         input_index=0,
        #         in_channels=48,
        #         feat_channels=48,
        #         num_classes=seg_classes,
        #         dropout_ratio=0.1,
        #         num_convs=2,
        #         bn_kwargs=bn_kwargs,
        #         int8_output=False,
        #     ),
        #     target=dict(
        #         type="FCNTarget",
        #     ),
        #     loss=dict(
        #         type="CrossEntropyLoss",
        #         loss_name="seg",
        #         reduction="mean",
        #         ignore_index=-1,
        #         use_sigmoid=use_bce,
        #         class_weight=2.0 if use_bce else [1.0, 5.0, 5.0, 5.0],
        #     ),
        #     decoder=dict(
        #         type="FCNDecoder",
        #         upsample_output_scale=1,
        #         use_bce=use_bce,
        #         bg_cls=-1,
        #     ),
        # ),
        dict(
            type="BevDetDecoder",
            name="bev_det",
            task_weight=1.0,
            head=dict(
                type="DepthwiseSeparableCenterPointHead",
                in_channels=48,
                tasks=tasks,
                share_conv_channels=48,
                share_conv_num=1,
                common_heads=dict(
                    reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2),
                ),
                head_conv_channels=48,
                num_heatmap_convs=2,
                final_kernel=3,
            ),
            target=dict(
                type="CenterPointTarget_Fisheye",
                class_names=tuple(CLASSES),
                tasks=tasks,
                gaussian_overlap=0.1,
                min_radius=2,
                out_size_factor=1,
                norm_bbox=True,
                max_num=500,
                bbox_weight=[1.0 for _ in range(8)] + [0.2, 0.2],
            ),
            loss_cls=dict(type="GaussianFocalLoss", loss_weight=1.0),
            loss_reg=dict(
                type="L1Loss",
                loss_weight=0.25,
            ),
            decoder=dict(
                type="CenterPointDecoder",
                class_names=tuple(CLASSES),
                tasks=tasks,
                bev_size=bev_size,
                out_size_factor=1,
                score_threshold=0.1,
                use_max_pool=True,
                nms_type=[
                    "rotate",
                    "rotate",
                    "rotate",
                    "circle",
                    "rotate",
                    "rotate",
                ],
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                nms_threshold=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
                decode_to_ego=True,
            ),
        ),
    ],
)

deploy_model = dict(
    type="ViewFusion",
    bev_feat_index=-1,
    bev_upscale=2,
    compile_model=True,
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type=efficientnet_TYPE,
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    neck=dict(
        type="FastSCNNNeck_Fisheye",
        in_channels=[112, 320],
        feat_channels=[64, 64],
        indexes=[-2, -1],
        bn_kwargs=bn_kwargs,
        scale_factor=2,
    ),
    view_transformer=view_transform,
    bev_encoder=dict(
        type="BevEncoder",
        backbone=dict(
            type="efficientnet",
            bn_kwargs=bn_kwargs,
            model_type=efficientnet_TYPE,
            num_classes=1000,
            include_top=False,
            activation="relu",
            use_se_block=False,
            input_channels=64,
            quant_input=False,
        ),
        neck=dict(
            type="BiFPN",
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[2, 4, 8, 16, 32],
            stride2channels=dict({2: 16, 4: 24, 8: 40, 16: 112, 32: 320}),
            out_channels=48,
            num_outs=5,
            stack=3,
            start_level=0,
            end_level=-1,
            fpn_name="bifpn_sum",
            upsample_type="function",
            use_fx=True,
        ),
    ),
    bev_decoders=[
        # dict(
        #     type="BevSegDecoder",
        #     name="bev_seg",
        #     use_bce=use_bce,
        #     bev_size=bev_size,
        #     task_size=task_map_size,
        #     grid_quant_scale=map_grid_quant_scale,
        #     task_weight=10.0,
        #     head=dict(
        #         type="DepthwiseSeparableFCNHead",
        #         input_index=0,
        #         in_channels=48,
        #         feat_channels=48,
        #         num_classes=seg_classes,
        #         dropout_ratio=0.1,
        #         num_convs=2,
        #         bn_kwargs=bn_kwargs,
        #         int8_output=False,
        #     ),
        # ),
        dict(
            type="BevDetDecoder",
            name="bev_det",
            task_weight=1.0,
            head=dict(
                type="DepthwiseSeparableCenterPointHead",
                in_channels=48,
                tasks=tasks,
                share_conv_channels=48,
                share_conv_num=1,
                common_heads=dict(
                    reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2),
                ),
                head_conv_channels=48,
                num_heatmap_convs=2,
                final_kernel=3,
            ),
        ),
    ],
)


W = int(grid_size[1])
H = int(grid_size[0])


def get_input_point(num_points):
    inputs = {"img": torch.randn((4,) + data_shape)}
    if original_gridsample:
        inputs["points0"] = torch.randn((num_points, H, W, 2))

        inputs["points1"] = torch.randn((num_points, H, W, 2))
    else:
        inputs["points0"] = torch.rand((num_points, feat_channels, grid_size[0] * grid_size[1])) * (1 << 10)    # torch.Size([10, 64, 16384])
        inputs["points1"] = torch.rand((num_points, 1, grid_size[0] * grid_size[1])) * (1 << 10)                # torch.Size([10, 1, 16384])

        inputs["points0"] = inputs["points0"].int()
        inputs["points1"] = inputs["points1"].int()

    return inputs


deploy_inputs = get_input_point(num_points)

# data
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="FisheyeBevDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        transforms=[
            dict(type="MultiViewsImgResize", size=resize_shape[1:]),
            dict(type="MultiViewsImgCrop", size=data_shape[1:]),
            dict(type="MultiViewsImgFlip", prob=0.5),
            dict(
                type="MultiViewsImgTransformWrapper",
                transforms=[
                    dict(type="PILToTensor"),
                    # dict(type="BgrToYuv444", rgb_input=True),
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
        bev_size=bev_size,
        map_size=map_size,
        map_path=meta_rootdir,
        with_bev_mask = False,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_nuscenesFisheye,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="FisheyeBevDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        transforms=[
            dict(type="MultiViewsImgResize", size=resize_shape[1:]),
            dict(type="MultiViewsImgCrop", size=data_shape[1:]),
            dict(
                type="MultiViewsImgTransformWrapper",
                transforms=[
                    dict(type="PILToTensor"),
                    # dict(type="BgrToYuv444", rgb_input=True),
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
        bev_size=bev_size,
        map_size=map_size,
        map_path=meta_rootdir,
        with_bev_mask=False,
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_nuscenesFisheye,
    sampler=None, 
    #dict(type=torch.utils.data.DistributedSampler),
)


def get_example_inputs(num_points):
    inputs = {"img": torch.randn((4,) + data_shape)}
    inputs["lidar2cam"] = torch.randn(
        (4, 4, 4),
    ).double()

    return inputs


example_inputs = get_example_inputs(num_points)


def get_qconfig_setter(is_calibration=True):
    qconfig_setter = QconfigSetter(
        reference_qconfig=get_qconfig(
            observer=(
                observer_v2.MSEObserver
                if is_calibration is True
                else observer_v2.MinMaxObserver
            ),
            fix_scale=False if is_calibration else True,
        ),
        templates=[
            ModuleNameTemplate({"": qint8}),
            ModuleNameTemplate(
                {"backbone.quant": {"dtype": qint8, "threshold": 1.0}},
            ),
            MatmulDtypeTemplate(
                input_dtypes=[qint8, qint8],
                prefix=None,
            ),
            ConvDtypeTemplate(
                input_dtype=qint8,
                weight_dtype=qint8,
                prefix=None,
            ),
        ],
        enable_optimize=True,
        save_dir="./qconfig_setting",
        custom_qconfig_mapping=None,
    )
    return qconfig_setter


calibration_qconfig_setter = get_qconfig_setter(True)
qat_qconfig_setter = get_qconfig_setter(False)

bpu_preprocess_data = preprocess_data
bpu_val_dataset = dict(
    type="FisheyeBevDataset",
    data_path=os.path.join(data_rootdir, "val_lmdb"),
    transforms=[
        dict(type="MultiViewsImgResize", size=resize_shape[1:]),
        dict(type="MultiViewsImgCrop", size=data_shape[1:]),
    ],
    bev_size=bev_size,
    map_size=map_size,
    map_path=meta_rootdir,
    with_bev_mask=False,
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=copy.deepcopy(qat_qconfig_setter),
            example_inputs=deploy_inputs,
        ),
    ],
)

export_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=copy.deepcopy(qat_qconfig_setter),
            example_inputs=deploy_inputs,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            # ignore_extra=True,
        ),
    ],
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs[1].items():
        losses.append(loss)
    return losses


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    enable_amp=enable_amp,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)

# val_nuscenes_metric = dict(
#     type="NuscenesMetric",
#     data_root=meta_rootdir,
#     version="v1.0-trainval",
#     save_prefix="./WORKSPACE/results" + task_name,
# )
val_nuscenes_metric=dict(
    type="Fcos3dGOYUMultiCamMetric",
    data_root=meta_rootdir,
    version="v1.0-trainval",
),

val_miou_metric = MeanIOU(seg_class=seg_classes_name, ignore_index=-1)
bpu_eval_metric = (val_nuscenes_metric, val_miou_metric)
bpu_process_result_data = process_bev_data
bpu_process_metric = process_bev_metric


def update_val_metric(metrics, batch, model_outs):
    # Convert one hot to inde
    preds = model_outs[1]["bev_det"]
    metrics[0].update(batch, preds)

    target: Tensor = batch["bev_seg_indices"]
    preds = model_outs[1]["bev_seg"]
    if use_bce is True:
        preds += 1
    metrics[1].update(target, preds)


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs[1])


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix="",
)

def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(batch, model_outs)

val_metric_updater = dict(
    type="MetricUpdater",
    # metrics=[val_nuscenes_metric, val_miou_metric],
    # metric_update_func=update_val_metric,
    metric_update_func=update_metric,
    step_log_freq=10000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

val_callback = dict(
    type="Validation",
    val_interval=1,
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=False,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=True,
    mode="max",
    monitor_metric_key="NDS",
)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    # type = "Trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                # checkpoint_path=os.path.join(predir, "model", 'ckpt', "float-checkpoint-best333.pth.tar"),
                # checkpoint_path=r'/home/Desktop/zoujiu/horizon/model/ckpt/float-checkpoint-best333.pth.tar',
                checkpoint_path=os.path.join(ckpt_dir, "float-checkpoint-best.pth.tar"),
                allow_miss=True,
                ignore_extra=True,
                check_hash=True,
                ignore_tensor_shape=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=weight_decay)},
        lr=start_lr,
    ),
    resume_optimizer = True,
    resume_epoch_or_step = True,
    resume_dataloader = True,
    batch_processor=batch_processor,
    device=None,
    num_epochs=train_epochs,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CyclicLrUpdater",
            target_ratio=(10, 1e-4),
            cyclic_times=1,
            step_ratio_up=0.4,
            step_log_interval=500,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="Fcos3dGOYUMultiCamMetric",
        data_root=meta_rootdir,
        version="v1.0-trainval",
    ),
)

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
if 'sampler' in calibration_data_loader.keys():
    calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 2
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 10

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                verbose=True,
            ),
            dict(
                type="Float2Calibration",
                convert_mode=convert_mode,
                qconfig_setter=copy.deepcopy(calibration_qconfig_setter),
                example_inputs=example_inputs,
            ),
            dict(type="DisableSyncConverter"),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        val_callback,
        ckpt_callback,
    ],
    log_interval=calibration_step / 10,
    val_metrics=dict(
        type="Fcos3dGOYUMultiCamMetric",
        data_root=meta_rootdir,
        version="v1.0-trainval",
    ),
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        # qconfig_params=dict(
        #     activation_qat_qkwargs=dict(
        #         averaging_constant=0,
        #     ),
        #     weight_qat_qkwargs=dict(
        #         averaging_constant=1,
        #     ),
        # ),
        converters=[
            dict(
                type="Float2QAT",
                convert_mode="jit-strip",
                # example_data_loader=copy.deepcopy(calibration_data_loader),
                example_inputs=example_inputs,
                qconfig_setter=qat_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    # ckpt_dir, "calibration-checkpoint-last.pth.tar"
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    resume_optimizer = True,
    resume_epoch_or_step = True,
    resume_dataloader = True,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=weight_decay)},
        lr=1e-6,
    ),
    batch_processor=batch_processor,
    num_epochs=qat_train_epochs,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        # dict(
        #     type="CyclicLrUpdater",
        #     target_ratio=(10, 1e-4),
        #     cyclic_times=1,
        #     step_ratio_up=0.4,
        #     step_log_interval=500,
        # ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="Fcos3dGOYUMultiCamMetric",
        data_root=meta_rootdir,
        version="v1.0-trainval",
    ),
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name + "_model",
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source="pyramid,ddr,ddr",
    opt="O2",
    transpose_dim=dict(
        inputs={
            "1": [0, 2, 3, 1],
            "2": [0, 2, 3, 1],
        }
    ),
    split_dim=dict(
        inputs={
            "0": [0, 6],
        }
    ),
)

float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Fcos3dGOYUMultiCamMetric",
        data_root=meta_rootdir,
        version="v1.0-trainval",
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

calibration_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2Calibration",
                convert_mode=convert_mode,
                qconfig_setter=copy.deepcopy(calibration_qconfig_setter),
                example_inputs=example_inputs,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="SetQuantRoundingMode"),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric, val_miou_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

qat_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2QAT",
                convert_mode="jit-strip",
                # example_data_loader=copy.deepcopy(calibration_data_loader),
                example_inputs=deploy_inputs,
                qconfig_setter=qat_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
            # dict(type="SetQuantRoundingMode"),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Fcos3dGOYUMultiCamMetric",
        data_root=meta_rootdir,
        version="v1.0-trainval",
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

infer_transforms = [
    dict(type="MultiViewsImgResize", size=resize_shape[1:]),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            # dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
]

hbir_deploy_model = copy.deepcopy(deploy_model)
hbir_deploy_model["compile_model"] = False

hbir_infer_model = dict(
    type="ViewFusionIrInfer",
    deploy_model=hbir_deploy_model,
    vt_input_hw=vt_input_hw,
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
    ir_model=dict(
        type="HbirModuleFisheye",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    bev_decoder_infers=[
        # dict(
        #     type="BevSegDecoderInfer",
        #     name="bev_seg",
        #     decoder=dict(
        #         type="FCNDecoder",
        #         upsample_output_scale=1,
        #         use_bce=use_bce,
        #         bg_cls=-1,
        #     ),
        # ),
        dict(
            type="BevDetDecoderInfer",
            name="bev_det",
            tasks=tasks,
            task_keys=["reg", "height", "dim", "rot", "vel", "heatmap"],
            decoder=dict(
                type="CenterPointDecoder",
                class_names=tuple(CLASSES),
                tasks=tasks,
                bev_size=bev_size,
                out_size_factor=1,
                score_threshold=0.1,
                use_max_pool=True,
                nms_type=[
                    "rotate",
                    "rotate",
                    "rotate",
                    "circle",
                    "rotate",
                    "rotate",
                ],
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                nms_threshold=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
                decode_to_ego=True,
            ),
        ),
    ],
)

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False

int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric, val_miou_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesFromImage",
        src_data_dir="./tmp_orig_data/nuscenes",
        version="v1.0-trainval",
        split_name="val",
        transforms=infer_transforms,
        bev_size=bev_size,
        map_size=map_size,
        map_path=meta_rootdir,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenesFisheye,
)

align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=align_bpu_data_loader,
    metrics=[val_nuscenes_metric, val_miou_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def resize_homo(homo, scale):
    view = np.eye(4)
    view[0, 0] = scale[1]
    view[1, 1] = scale[0]
    homo = view @ homo
    return homo


def crop_homo(homo, offset):
    view = np.eye(4)
    view[0, 2] = -offset[0]
    view[1, 2] = -offset[1]
    homo = view @ homo
    return homo

def process_img(img_path, resize_size, crop_size):
    orig_img = cv2.imread(img_path)
    orig_img = Image.fromarray(orig_img)
    orig_img = pil_to_tensor(orig_img)

    resize_hw = (
        int(resize_size[0]),
        int(resize_size[1]),
    )

    orig_shape = (orig_img.shape[1], orig_img.shape[2])
    resized_img = resize(orig_img, resize_hw).unsqueeze(0)
    top = int(resize_hw[0] - crop_size[0])
    left = int((resize_hw[1] - crop_size[1]) / 2)
    resized_img = resized_img[:, :, top:, left:]
    if isinstance(orig_img, np.ndarray):
        resized_img = torch.from_numpy(orig_img).unsqueeze(0)
    else:
        resized_img = orig_img.unsqueeze(0)

    return resized_img, orig_shape

def process_inputs(infer_inputs, transforms=None):
    cam_names = [
        "fisheye_camera_front",
        "fisheye_camera_left",
        "fisheye_camera_rear",
        "fisheye_camera_right"
    ]
    resize_size = resize_shape[1:]
    input_size = val_data_shape[1:]
    orig_imgs = []
    file_list = list(os.listdir(infer_inputs))
    image_dir_list = list(filter(lambda x: x.endswith(".jpg") or x.endswith(".png"), file_list))
    image_dir_list.sort()
    for i, img in enumerate(image_dir_list):
        name = copy.deepcopy(img)
        if cam_names[i] not in img:
            exit(-1)
        img = os.path.join(infer_inputs, img)
        img, orig_shape = process_img(img, resize_size, input_size)
        orig_imgs.append({"name": name, "img": img})

    input_imgs = []
    for orig_img in orig_imgs:
        # input_img = horizon.nn.functional.bgr_to_yuv444(orig_img["img"], True)
        input_img = orig_img["img"].clone()
        input_imgs.append(input_img)

    input_imgs = torch.cat(input_imgs)
    input_imgs = (input_imgs - 128.0) / 128.0

    homo = np.load(os.path.join(infer_inputs, f"{homo_key}.npy"))

    # top = int(resize_size[0] - input_size[0])
    # left = int((resize_size[1] - input_size[1]) / 2)

    # scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    # homo = resize_homo(homo, scale)
    # homo = crop_homo(homo, (left, top))

    model_input = {
        "img": input_imgs,
        f"{homo_key}": torch.tensor(homo),
    }
    # if transforms is not None:
        # model_input = transforms(model_input)

    vis_inputs = {}
    vis_inputs["img"] = orig_imgs
    vis_inputs["meta"] = {f"{homo_key}": homo}

    return model_input, vis_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs[1]

    viz_func(vis_inputs["img"], preds, vis_inputs["meta"])
    return None


single_infer_dataset = copy.deepcopy(align_bpu_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    for image_idx, (img_name, img_data) in enumerate(
        zip(data["img_name"], data["img"])
    ):
        save_name = f"img{image_idx}_{os.path.basename(img_name)}"
        img_data.save(os.path.join(save_path, save_name), "JPEG")

    ego2img_path = os.path.join(save_path, "ego2img.npy")
    np.save(ego2img_path, np.array(data["ego2img"]))

infer_float_cfg = dict(
    model=model,
    stage="float",
    # os.path.join(ckpt_dir, "float-checkpoint-best.pth.tar"),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=(
                    "/home/Desktop/fisheyedod/horizon/ckptsave/float-checkpoint-best.pth.tar"  # noqa
                    # "/home/Desktop/tmp_models/fcos3d_efficientnetb0_nuscenes/float-checkpoint-epoch-0015-2c10799b.pth.tar"  # noqa
                ),
                allow_miss=True,
                # ignore_tensor_shape=True,
                # ignore_extra=True,
            ),
        ],
    ),
    # gen_inputs_cfg=dict(
    #     dataset=single_infer_dataset,
    #     sample_idx=[0],
    #     inputs_save_func=inputs_save_func,
    # ),
    process_inputs=process_inputs,
    # transforms=infer_transforms,
    # viz_func=dict(type="Cam3dViz"),
    viz_func=dict(
        type="NuscenesViz_Fisheye", is_plot=True, 
        ocam_path=ocam_path, cam_fov=cam_fov,
        bev_size=bev_size, use_bce=use_bce
    ),
    process_outputs=process_outputs,
)

infer_cfg = dict(
    model=hbir_infer_model,
    input_path="./demo/bev_lss_efficientnetb0_multitask_nuscenes",
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[0],
        inputs_save_func=inputs_save_func,
    ),
    process_inputs=process_inputs,
    viz_func=dict(
        type="NuscenesViz_Fisheye", is_plot=True,
        ocam_path=ocam_path, cam_fov=cam_fov,
        bev_size=bev_size, use_bce=use_bce
    ),
    process_outputs=process_outputs,
)

onnx_cfg = dict(
    model=deploy_model,
    inputs=get_input_point(num_points),
    stage="qat",
    kwargs=dict(
        input_names=['images'],
    ),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
                qconfig_setter=copy.deepcopy(qat_qconfig_setter),
                example_inputs=deploy_inputs,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
)


onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    kwargs=dict(
        input_names=['images'],
        opset_version=16,
    ),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                allow_miss = True,
                # ignore_extra = False,
                # ignore_tensor_shape = False,
            ),
        ],
    ),
)


# onnx_cfg = dict(
#     model=deploy_model,
#     stage="calibration",
#     inputs=get_input_point(num_points),
#     kwargs=dict(
#         input_names=['images', 'points0', 'points1'],
#     ),
#     model_convert_pipeline=dict(
#         type="ModelConvertPipeline",
#         qat_mode="fuse_bn",
#         converters=[
#             dict(
#                 type="Float2Calibration",
#                 convert_mode=convert_mode,
#                 qconfig_setter=copy.deepcopy(calibration_qconfig_setter),
#                 example_inputs=deploy_inputs,
#             ),
#             dict(
#                 type="LoadCheckpoint",
#                 checkpoint_path=os.path.join(
#                     ckpt_dir, "calibration-checkpoint-last.pth.tar"
#                 ),
#             ),
#             dict(type="SetQuantRoundingMode"),
#         ],
#     ),
# )
