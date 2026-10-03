"""align bpu validation tools, Only support int-infer."""

import os
import cv2
import copy
import torch
import shutil
import argparse
import numpy as np
from PIL import Image
from hat.utils.config import Config
from hat.utils.logger import init_logger
import sys
sys.path.append(r'/home/Desktop/fisheyedod/horizon')
from lmdbdata.fisheye_carla_dataset import *
from model.view_transformerFisheye import *
from model.fcos3d_goyu_metric import *
from model.fisheye_lss import *
from ocamcamera import OcamCamera
import horizon_plugin_pytorch as horizon
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion
import mmengine
from hat.registry import RegistryContext, build_from_registry
from torchvision.transforms.functional import pil_to_tensor
try:
    from torchvision.transforms.functional_tensor import resize
except ImportError:
    # torchvision 0.18
    from torchvision.transforms._functional_tensor import resize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=False,
        default=r'/home/Desktop/fortrain/horizon/config/bev_lss_efficientnetb0_multitask_nuscenes.py',
        help="train config file path",
    )
    parser.add_argument('--savePath', default=r'/home/Desktop/fisheyedod/horizon/infer_out')
    return parser.parse_args()

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

if __name__ == "__main__":
    args = parse_args()

    config = Config.fromfile(args.config)
    init_logger(f".hat_logs/{config.task_name}_infer_viz")

    homo_key = 'lidar2cam'
    resize_shape = (3, 396, 704)
    val_data_shape = (3, 256, 704)

    # config._cfg_dict['val_data_loader']['dataset']['transforms'] = None
    bevSize = config._cfg_dict['bev_size']
    config._cfg_dict['val_data_loader']['dataset']['transforms'] = [
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
    config._cfg_dict['val_data_loader']['batch_size'] = 1
    config._cfg_dict['num_workers'] = 1
    config._cfg_dict['shuffle'] = True

    # deploy_model = build_from_registry(config.get("deploy_model"))
    # camviz = Cam3dViz()

    for i in os.listdir(args.savePath):
        pth = os.path.join(args.savePath, i)
        if os.path.isdir(pth):
            shutil.rmtree(pth)
        else:
            os.remove(pth)
    cam_fov = 220
    ocam_path='/home/Desktop/fisheyedod/horizon/lmdbdata/calib_results.txt'
    omni_ocam = OcamCamera(filename=ocam_path, fov=cam_fov)
    use_box_not_camviz = 'nuscene'
    out_dir = args.savePath

    # step 1
    allfile = []
    sourdir = r'/home/data/Fisheye3DODdataset/ImageSets-2hz/fisheye3dod_infos_val.pkl'
    imgsource = r'/home/data/Fisheye3DODdataset'
    jsonfp = mmengine.load(sourdir)
    # for idx, data in enumerate(val_data_loader):
    #     allfile.append([data['img_name'][0], data['lidar2cam'].cpu().numpy()])
    kk = ['fisheye_camera_front', 'fisheye_camera_left', 'fisheye_camera_rear', 'fisheye_camera_right']
    kk.sort()
    source = r'/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ'
    if os.path.exists(source):
        shutil.rmtree(source)
    os.makedirs(source, exist_ok = True)
    choose = []
    for idx, datalist in enumerate(jsonfp['data_list']):
        pth = []
        l2c = []
        find = False
        for type in ['fisheye_camera_front', 'fisheye_camera_left', 'fisheye_camera_rear', 'fisheye_camera_right']:
            lidar2cam = datalist['cam_info']['cam_fisheye'][type]['lidar2cam']
            cam_path = datalist['cam_info']['cam_fisheye'][type]['cam_path']
            cam_path = os.path.join(imgsource, cam_path)
            pth.append(cam_path)
            l2c.append(lidar2cam)
        #     if "00848826.png" in cam_path and 'train-Town02_Opt-SoftRainNoon-2024_09_25_14_26_29' in cam_path:
        #         find = True
        # if find:
        #     choose.append([pth, np.array(l2c)])
        #     np.save(f"{source}/1/ego2img.npy", np.array(l2c))
        allfile.append( [ pth, np.array(l2c) ] )

    ch = 30
    chnow = np.random.randint(0, len(allfile), ch)
    for i in chnow:
        choose.append(allfile[i])

    num = 1
    for i in range(ch): # 1 2 3 4 5 6 7 9 10 11
        path, mat = choose[i]
        pt = f'{source}/{num}'
        if not os.path.exists(pt):
            os.makedirs(pt)
        else:
            num += 1
            continue
        for i in path:
            name = "_".join(i.split(os.sep)[-2:])
            shutil.copyfile(i, os.path.join(pt, name))
        np.save(f"{pt}/lidar2cam.npy", mat)
        num += 1

    # model = build_from_registry(config.get("model"))

    data_loader = build_from_registry(config.get("val_data_loader"))

    for idx, data in enumerate(data_loader):
        kk = 0
        break

    path = source
    for i in os.listdir(path):
        pth = os.path.join(path, i)
        model_input_col, vis_inputs_col = process_inputs(pth)
        images = model_input_col["img"]
        np.save(os.path.join(pth, "images.npy"), images)
        if not os.path.isdir(pth):
            continue
        try:
            _ = int(i)
        except:
            continue
        lidar2camPATH = os.path.join(pth, "lidar2cam.npy")
        lidar2cam = np.load(lidar2camPATH)
        lidar2cam = torch.from_numpy(lidar2cam).double().to("cuda:0")
        # model.view_transformer.homo_key = "lidar2cam"
        meta = { 'lidar2cam' : lidar2cam }
        feat_hw = config.get('feat_hw')
        # model = model.to('cuda:0')
        # points = model.view_transformer.export_reference_points(meta = meta, feat_hw = feat_hw)
        # np.save(os.path.join(pth, "points0Uh.npy"), points['points0'][0].cpu().numpy())
        # np.save(os.path.join(pth, "points0Vh.npy"), points['points0'][1].cpu().numpy())
        # np.save(os.path.join(pth, "points1Uh.npy"), points['points1'][0].cpu().numpy())
        # np.save(os.path.join(pth, "points1Vh.npy"), points['points1'][1].cpu().numpy())
        # points['points0'].cpu().numpy().tofile(os.path.join(pth, "points0.bin"))
        # points['points0'].cpu().numpy().tofile(os.path.join(pth, "points1.bin"))
    # camviz = Cam3dViz()
