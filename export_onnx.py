import os
import onnx
import torch
import onnxsim
import logging
import argparse
import numpy as np
import horizon_plugin_pytorch as horizon
from horizon_plugin_pytorch.quantization import (
    FakeQuantState,
    set_fake_quantize,
)
from horizon_plugin_pytorch.utils.onnx_helper import (
    export_quantized_onnx,
    export_to_onnx,
)
import onnx
import onnxsim
from onnx import helper
from hat.registry import RegistryContext, build_from_registry
from hat.utils.config import Config
from hat.utils.logger import MSGColor, format_msg
from hat.utils.setup_env import setup_args_env

from lmdbdata.fisheye_carla_dataset import *
from model.view_transformerFisheye import *
from model.fcos3d_goyu_metric import *
# from model.fisheye_lss import FisheyeLSSTransform
from model.fast_scnn import FastSCNNNeck_Fisheye
from model.multi_views import BevFeatureRotate_Fisheye
from model.target import CenterPointTarget_Fisheye

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)-15s %(levelname)s %(message)s",
    level=logging.INFO,
)

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
        default=r'/home/Desktop/fisheyedod/horizon/config/bev_lss_efficientnetb0_multitask_nuscenes.py',
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
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args

def nchw_to_nhwc_onnx(input_onnx_path, output_onnx_path):
    # 1. 加载模型
    model = onnx.load(input_onnx_path)
    graph = model.graph
    outputs = graph.output

    # 遍历所有输出
    for idx, old_output in enumerate(outputs):
        out_name = old_output.name
        out_dtype = old_output.type.tensor_type.elem_type
        
        # 获取原始维度信息（保留动态维度，不读取值）
        old_shape = old_output.type.tensor_type.shape.dim
        
        # 2. 添加 Transpose 节点
        transpose_node = helper.make_node(
            "Transpose",
            inputs=[out_name],
            outputs=[f"{out_name}_nhwc"],
            perm=[0, 2, 3, 1],
            name=f"transpose_{idx}"
        )
        graph.node.append(transpose_node)
        
        # 3. 关键：正确构造 NHWC 动态形状（不填0，不填空）
        new_output = helper.make_tensor_value_info(
            f"{out_name}_nhwc",
            out_dtype,
            shape=["N", "H", "W", "C"]  # 动态维度写法，兼容所有推理引擎
        )
        # 替换旧输出
        graph.output.remove(old_output)
        graph.output.insert(idx, new_output)

    # 4. 安全简化模型（不会报错）
    try:
        from onnxsim import simplify
        model_simp, check = simplify(model, skip_shape_inference=True)  # 核心：跳过形状推断
        if check:
            model = model_simp
            print("模型简化成功！")
    except Exception as e:
        print(f"简化跳过（不影响使用）: {e}")

    # 保存
    onnx.save(model, output_onnx_path)
    print(f"✅ 转换成功！保存至: {output_onnx_path}")
    print(f"✅ 输出已改为 NHWC 格式")

if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    cfg = Config.fromfile(args.config)

    logger.info("=" * 50 + "BEGIN EXPORT ONNX" + "=" * 50)

    if "march" not in cfg:
        logger.warning(
            format_msg(
                f"Please make sure the march is provided in configs. "
                f"Defaultly use {horizon.march.March.BAYES}",
                MSGColor.RED,
            )
        )
    horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))

    with RegistryContext():
        onnx_cfg_ = cfg.get('onnx_cfg')
        onnx_cfg_['model']['view_transformer']['learning_point_version'] = str(args.learning_point_version)
        onnx_cfg_['model']['view_transformer']['ocam_path'] = str(args.ocam_path)
        onnx_cfg_['model_convert_pipeline']['converters'][0]['checkpoint_path'] = args.ckpt
        if int(args.learning_point_version) >= 9:
            onnx_cfg_['model']['view_transformer']['hidden_matmul'] = onnx_cfg_['model']['view_transformer']['grid_size'][0]
        else:
            onnx_cfg_['model']['view_transformer']['hidden_matmul'] = 256
        onnx_solver = build_from_registry(onnx_cfg_)
        # data_loader = build_from_registry(cfg.get("data_loader"))
        # for idx, data in enumerate(data_loader):
        #     break
        model = onnx_solver["model"]
        # meta = { 'lidar2cam':  }
        # feat_wh = [50, 50]

        # ref_p_dict = model.view_transformer.export_reference_points(meta, feat_wh)
        # model.view_transformer.ref_point = (ref_p_dict['point0'], ref_p_dict['point1'])

        pipeline = onnx_solver.get("model_convert_pipeline")
        stage = onnx_solver["stage"]
        if pipeline is not None:
            model = pipeline(model)
        else:
            logger.warning(
                format_msg(
                    f"not define model_convert_pipeline for {stage} stage "
                    f"model, will directly export the model to onnx...",
                    MSGColor.RED,
                )
            )
        model = model.eval()

    example_input = onnx_solver.get("inputs", cfg.deploy_inputs)
    out_dir = onnx_solver.get("out_dir", cfg.get("ckpt_dir", "."))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    file_path = os.path.join(out_dir, stage + ".onnx")
    kwargs = onnx_solver.get("kwargs", {})

    logger.info("will export {} model to onnx...".format(stage))
    if stage == "int_infer":
        # If a dictionary is the last element of the args tuple, it will be
        # interpreted as containing named arguments. In order to pass a dict as
        # the last non-keyword arg, provide an empty dict as the last element
        # of the args tuple.
        export_quantized_onnx(model, (example_input, {}), file_path, **kwargs)
    elif stage == "qat":
        set_fake_quantize(model.eval(), FakeQuantState.VALIDATION)
        export_to_onnx(model, (example_input, {}), file_path, **kwargs)
    else:
        # z0 = torch.rand((10, 64, 128 * 128)) * (1 << 10)    # torch.Size([10, 64, 16384])
        # z1 = torch.rand((10, 1, 128 * 128)) * (1 << 10)                # torch.Size([10, 1, 16384])
        # z0U = np.load(r'/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ/9/points0Uh.npy')
        # z0V = np.load(r'/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ/9/points0Vh.npy')
        # z1U = np.load(r'/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ/9/points1Uh.npy')
        # z1V = np.load(r'/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ/9/points1Vh.npy')
        # z0u = torch.from_numpy(z0U)#.int()#[0].unsqueeze(0)
        # z0v = torch.from_numpy(z0V)#.int()#[0].unsqueeze(0)
        # z1u = torch.from_numpy(z1U)#.int()#[0].unsqueeze(0)
        # z1v = torch.from_numpy(z1V)#.int()#[0].unsqueeze(0)
        # model.view_transformer.ref_point = [(z0u, z0v), (z1u, z1v)]
        torch.onnx.export(model, (example_input, {}), file_path, **kwargs)

        nchw2nhwc = False
        if nchw2nhwc:
            nchw_to_nhwc_onnx(file_path, file_path.replace(".onnx", "_nhwc.onnx"))
            file_path = file_path.replace(".onnx", "_nhwc.onnx")
        model = onnx.load(file_path)
        sim, check = onnxsim.simplify(model, check_n=3)
        onnx.save(sim, file_path)

    logger.info("=" * 50 + "END ONNX" + "=" * 50)
