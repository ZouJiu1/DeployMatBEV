"""Pack nuscenes."""

import argparse
import os
import sys
from multiprocessing import cpu_count
fileAbspath = __file__
fileName = os.path.basename(__file__)
abspath = fileAbspath.replace(fileName, "")
predir = os.path.abspath(os.path.join(abspath, ".."))

sys.path.append(predir)
from lmdbdata.fisheye_carla_dataset import fisheyeGOYUPacker
from hat.utils.logger import init_logger

cpu_num = cpu_count()

def parse_args():
    parser = argparse.ArgumentParser(description="Pack nuscenes dataset.")
    parser.add_argument(
        "--src-data-dir",
        default=r"/media/datasets/Fisheye3DODdataset",
        # default=r"/home/data/Fisheye3DODdataset",
        required=False,
        help="The directory that contains unpacked image files.",
    )
    parser.add_argument(
        "--meta_json_dir",
        default=r'/media/datasets/Fisheye3DODdataset/ImageSets-2hz',
        # default=r'/home/data/Fisheye3DODdataset/ImageSets-2hz',
        required=False,
        help=""
    )
    parser.add_argument(
        "--pack-type",
        default=r'lmdb',
        required=False,
        help="The pack data type for result of packer",
    )
    parser.add_argument(
        "--target-data-dir",
        default=r"/media/datasets/Fisheye3DODdataset/lmdb",
        # default=r"/home/data/Fisheye3DODdataset/lmdb",
        help="The directory for result of packer",
    )
    parser.add_argument(
        "--split-name", default="train", choices=['train', 'val'], help="The split to paccked."
    )
    parser.add_argument(
        "--exclude_dirs", default=['lmdb', 'v1.0-trainval'], \
        help="The split to paccked.", type = float,
    )
    parser.add_argument(
        "--num-workers",
        default=-1,
        help="The number of workers to load image.",
    )
    # parser.add_argument(
    #     '--camera_type',
    #     default=None,
    #     type = list,
    #     help = "",
    # )
    # parser.add_argument(
    #     '--choose_camera',
    #     default='camera_front',
    #     type = str,
    #     help = "",
    # )
    parser.add_argument(
        "--only-lidar",
        action="store_true",
    )
    parser.add_argument(
        "--need-occ",
        action="store_true",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    init_logger(".hat_logs/nuscenes_packer")
    directory = args.src_data_dir
    print("Loading dataset from %s" % directory)

    if args.target_data_dir == "":
        args.target_data_dir = args.src_data_dir
    target_data_dir = os.path.join(
        args.target_data_dir,
        "%s_%s" % (args.split_name, args.pack_type),
    )
    # if args.choose_camera not in args.camera_type:
    #     exit(-1)
    packer = fisheyeGOYUPacker(
        target_data_dir = target_data_dir,
        meta_json_dir = args.meta_json_dir,
        exclude_dirs = args.exclude_dirs,
        src_data_dir = args.src_data_dir,
        num_workers = int(args.num_workers),
        pack_type = args.pack_type,
        only_lidar = args.only_lidar,
        need_occ=args.need_occ,
        split_name = args.split_name,
        camera_type = None,
        choose_camera = None,
    )
    packer()
