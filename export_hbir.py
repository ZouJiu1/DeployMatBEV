# Copyright (c) Horizon Robotics. All rights reserved.
# Export hbir from QAT model, and convert QAT hbir to Quantized hbir.
# This script requires user define key "hbir_exporter" in config file, whose
# type must be "HbirExporter".
# If "hbir_exporter" is not defined, we will try to gather required info
# from "deploy_model", "deploy_inputs", "qat/calibration_predictor"
# and "ckpt_dir".

import argparse
import logging

import horizon_plugin_pytorch as horizon

from hat.registry import RegistryContext, build_from_registry
from hat.utils.config import Config
from hat.utils.hbdk4.hbir_exporter import HbirExporter
from hat.utils.logger import MSGColor, format_msg
from hat.utils.setup_env import setup_args_env

from lmdbdata.fisheye_carla_dataset import *
from model.view_transformerFisheye import *
from model.fcos3d_goyu_metric import *
from model.fisheye_lss import FisheyeLSSTransform
from model.fast_scnn import FastSCNNNeck_Fisheye
from model.multi_views import BevFeatureRotate_Fisheye
from model.target import CenterPointTarget_Fisheye

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=False,
        default=r'/home/Desktop/fisheyedod/horizon/config/bev_lss_efficientnetb0_multitask_nuscenes.py',
        help="Config file path.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=r'/home/Desktop/fisheyedod/horizon/ckptsave',
        required=False,
        help="Where to save exported hbir files.",
    )
    parser.add_argument(
        "--with-check",
        action="store_true",
        help="Whether check the mlir model forward with example input.",
    )
    parser.add_argument(
        "--debug",
        # default = True,
        default = False,
        # action="store_true",
        help="Set logger level to DEBUG."
    )
    parser.add_argument(
        "--march",
        "-m",
        type=str,
        default=None,
        required=False,
        help="BPU march.",
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def export_hbir(
    config_file_path,
    save_path=None,
    with_check=False,
    debug=False,
    march=None,
):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s",
        level=logging.DEBUG if debug else logging.INFO,
    )

    cfg = Config.fromfile(config_file_path)

    logger.info("=" * 50 + "BEGIN EXPORTING HBIR" + "=" * 50)

    if "hbir_exporter" in cfg:
        hbir_exporter = cfg.hbir_exporter
        if (
            "type" not in hbir_exporter
            or hbir_exporter["type"] != "HbirExporter"
        ):
            msg = "hbir_exporter must have 'type'='HbirExporter'"
            logger.error(msg)
            raise ValueError(msg)
    else:
        logger.warning(
            format_msg(
                "Do not find hbir_exporter in config file, "
                "trying to collect need info automaticlly.",
                MSGColor.RED,
            )
        )

        def raise_key_missing_error(key):
            msg = "Do not find {} in config file.".format(key)
            logger.error(format_msg(msg, MSGColor.RED))
            raise ValueError(msg)

        if "deploy_model" in cfg:
            model = cfg.deploy_model
        else:
            raise_key_missing_error("deploy_model")

        if "deploy_inputs" in cfg:
            example_inputs = cfg.deploy_inputs
        else:
            raise_key_missing_error("deploy_inputs")

        if "export_model_convert_pipeline" in cfg:
            model_convert_pipeline = cfg.export_model_convert_pipeline
        else:
            if "qat_predictor" in cfg:
                predictor = cfg.qat_predictor
            elif "calibration_predictor" in cfg:
                predictor = cfg.calibration_predictor
            else:
                msg = (
                    "Do not find qat_predictor or calibration_predictor"
                    " in config file."
                )
                logger.error(format_msg(msg, MSGColor.RED))
                raise ValueError(msg)
            model_convert_pipeline = predictor["model_convert_pipeline"]
        if "hbir_save_dir" in cfg:
            cfg_save_path = cfg.hbir_save_dir
        elif "ckpt_dir" in cfg:
            cfg_save_path = cfg.ckpt_dir
        else:
            raise_key_missing_error("ckpt_dir")
        if "output_names" in cfg:
            output_names = cfg.output_names
        else:
            output_names = None

        hbir_exporter = dict(  # noqa: C408
            type="HbirExporter",
            model=model,
            model_convert_pipeline=model_convert_pipeline,
            example_inputs=example_inputs,
            save_path=cfg_save_path,
            model_name=cfg.get("task_name", "model"),
            input_names=list(example_inputs.keys()),
            output_names=output_names,
            enable_vpu=cfg.get("enable_vpu", False),
            check_non_tensor_inout=False,
        )

    if "march" not in hbir_exporter:
        if "march" not in cfg:
            logger.warning(
                format_msg(
                    "Please make sure the march is provided in configs. "
                    "Defaultly use {}".format(horizon.march.March.NASH_E),
                    MSGColor.RED,
                )
            )

            hbir_exporter["march"] = horizon.march.March.NASH_E
        else:
            hbir_exporter["march"] = cfg.march
    if march is not None:
        hbir_exporter["march_modify"] = march
    if save_path is not None:
        hbir_exporter["save_path"] = save_path

    hbir_exporter["with_check"] = (
        hbir_exporter.get("with_check", False) or with_check
    )

    logger.info("Export config:")
    logger.info(horizon.utils.misc.tensor_struct_repr(hbir_exporter))

    with RegistryContext():
        hbir_exporter: HbirExporter = build_from_registry(hbir_exporter)
        hbir_exporter(debug)

    logger.info("=" * 50 + "END EXPORTING HBIR" + "=" * 50)


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    export_hbir(
        args.config, args.save_path, args.with_check, args.debug, args.march
    )
