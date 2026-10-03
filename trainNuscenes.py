"""train tools."""
import argparse
import ast
import logging
import os
import pprint
import warnings
from typing import List, Sequence, Union

import horizon_plugin_pytorch as horizon
import torch.backends.cudnn as cudnn

from hat.engine import build_launcher
from hat.registry import RegistryContext, build_from_registry
from hat.utils.config import Config, ConfigVersion, filter_configs
from hat.utils.distributed import get_dist_info, rank_zero_only
from hat.utils.logger import (
    LOG_DIR,
    DisableLogger,
    MSGColor,
    format_msg,
    init_rank_logger,
    rank_zero_info,
)
from hat.utils.seed import seed_training
from hat.utils.setup_env import setup_args_env, setup_hat_env
from hat.utils.thread_init import init_num_threads

from lmdbdata.fisheye_carla_dataset import *
from model.view_transformerFisheye import *
from model.fcos3d_goyu_metric import *
# from model.fisheye_lss import FisheyeLSSTransform
from model.fast_scnn import FastSCNNNeck_Fisheye
from model.multi_views import BevFeatureRotate_Fisheye
from model.target import CenterPointTarget_Fisheye

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

fileAbspath = __file__
fileName = os.path.basename(__file__)
abspath = fileAbspath.replace(fileName, "")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        "-s",
        type=str,
        default=r'float',
        # default=r'calibration',
        # default=r'qat',
        required=False,
        help=(
            "the training stage, you should define "
            "{stage}_trainer in your config"
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=os.path.join(abspath, r'config/nuscene_config.py'),
        required=False,
        help="train config file path",
    )
    parser.add_argument(
        "--device-ids",
        "-ids",
        type=str,
        required=False,
        default="0,1",
        help="GPU device ids like '0,1,2,3', "
        "will override `device_ids` in config",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="tcp://localhost:10001", # auto
        help="dist url for init process, such as tcp://localhost:8000",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["torch", "mpi"],
        default=None,
        help="job launcher for multi machines",
    )
    parser.add_argument(
        "--pipeline-test",
        action="store_true",
        default=False,
        help="export HAT_PIPELINE_TEST=1, which used in config",
    )
    parser.add_argument(
        "--opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--opts-overwrite",
        type=ast.literal_eval,
        default=True,
        help="True or False, default True, "
        "Weather to overwrite existing (keys, values) in configs",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=logging.WARNING,
        help="Set the logging level for other rank except rank0",
    )

    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def train_entrance(
    device: Union[None, int, Sequence[int]],
    cfg_file: str,
    cfg_opts: List,
    cfg_opts_overwrite: bool,
    stage: str,
    level: int = logging.WARNING,
):
    """Training entrance function for launcher.

    Args:
        device: run on cpu (if None), or gpu (list of gpu ids)
        cfg_file: Config file used to build log file name.
        cfg_opts: Custom config options from command-line.
        cfg_opts_overwrite: Weather to overwrite existing {k: v} in configs.
        stage: Current training stage used to build log file name.
        level: logging level for other rank except rank0.
    """
    cfg = Config.fromfile(cfg_file)
    if cfg_opts is not None:
        cfg.merge_from_list_or_dict(cfg_opts, overwrite=cfg_opts_overwrite)

    rank, world_size = get_dist_info()
    disable_logger = rank != 0 and cfg.get("log_rank_zero_only", False)

    # 1. init logger
    logger = init_rank_logger(
        rank,
        save_dir=cfg.get("log_dir", LOG_DIR),
        cfg_file=cfg_file,
        step=stage,
        prefix="train-",
    )

    if disable_logger:
        logger.info(
            format_msg(
                f"Logger of rank {rank} has been disable, turn off "
                "`log_rank_zero_only` in config if you don't want this.",
                MSGColor.GREEN,
            )
        )

    if (
        "redirect_config_logging_path" in cfg
        and cfg["redirect_config_logging_path"]
        and rank == 0
    ):
        with open(cfg["redirect_config_logging_path"], "w") as fid:
            fid.write(pprint.pformat(filter_configs(cfg)))
        rank_zero_info(
            "save config logging output to %s"
            % cfg["redirect_config_logging_path"]
        )
    else:
        rank_zero_info(pprint.pformat(filter_configs(cfg)))

    rank_zero_info("=" * 50 + "BEGIN %s STAGE" % stage.upper() + "=" * 50)
    # 2. init num threads
    init_num_threads()
    # 3. seed training
    cudnn.benchmark = cfg.cudnn_benchmark
    if cfg.seed is not None:
        seed_training(cfg.seed)

    if "march" not in cfg:
        rank_zero_only(
            logger.warning(
                format_msg(
                    f"Please make sure the march is provided in configs. "
                    f"Defaultly use {horizon.march.March.BAYES}",
                    MSGColor.RED,
                )
            )
        )
    horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))

    # 4. build and run trainer
    with DisableLogger(disable_logger, level), RegistryContext():
        trainer = getattr(cfg, f"{stage}_trainer")
        trainer["device"] = device
        trainer = build_from_registry(trainer)
        trainer.fit()

    rank_zero_info("=" * 50 + "END %s STAGE" % stage.upper() + "=" * 50)

def train(
    stage: str,
    config: str,
    device_ids: str = None,
    dist_url: str = "auto",
    launcher: str = None,
    pipeline_test: bool = False,
    opts: list = None,
    opts_overwrite: bool = None,
    args_env: list = None,
    level: int = logging.WARNING,
):
    """Training  function.

    Args:
        stage: The training stage, you should define {stage}_trainer
               in your config.
        config: Config file path.
        device_ids: GPU device ids like '0,1,2,3', will override
                    `device_ids` in config.
        dist_url: Dist url for init process, such as tcp://localhost:8000.
        launcher: Job launcher for multi machines.
        pipeline_test: export HAT_PIPELINE_TEST=1, which used in config.
        opts: modify config options.
        opts_overwrite: True or False, default True, weather to overwrite
                        existing (keys, values) in configs.
        args_env: The args will be set in env.
        level: logging level for other rank except rank0.
    """
    if args_env:
        setup_args_env(args_env)
    setup_hat_env(stage, pipeline_test)

    config_info = Config.fromfile(config)
    # check config version
    config_version = config_info.get("VERSION", None)
    if config_version is not None:
        assert (
            config_version == ConfigVersion.v2
        ), "{} only support config with version 2, not version {}".format(
            os.path.basename(__file__), config_version.value
        )
    else:
        warnings.warn(
            "VERSION will must set in config in the future."
            "You can refer to configs/classification/resnet18.py,"
            "and configs/classification/bernoulli/mobilenetv1.py."
        )

    if opts is not None:
        config_info.merge_from_list_or_dict(opts, overwrite=opts_overwrite)

    if device_ids is not None:
        ids = list(map(int, device_ids.split(",")))
    else:
        ids = config_info.device_ids

    assert hasattr(
        config_info, f"{stage}_trainer"
    ), f"There are not {stage}_trainer in config"
    trainer_config = getattr(config_info, f"{stage}_trainer")

    launch = build_launcher(trainer_config)
    launch(
        train_entrance,
        ids,
        dist_url=dist_url,
        dist_launcher=launcher,
        args=(
            config,
            opts,
            opts_overwrite,
            stage,
            level,
        ),
    )

if __name__ == "__main__":
    '''
    original_bevpool = False
    useLidar2cam = False
    useSpherical = True
    '''
    try:
        args, args_env = parse_args()

        train(
            stage=args.stage,
            config=args.config,
            device_ids=args.device_ids,
            dist_url=args.dist_url,
            launcher=args.launcher,
            pipeline_test=args.pipeline_test,
            opts=args.opts,
            opts_overwrite=args.opts_overwrite,
            level=args.level,
            args_env=args_env,
        )
    except Exception as e:
        logger.error("train failed! " + str(e))
        raise e
