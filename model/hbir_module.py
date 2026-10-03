# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Callable, Dict, Optional, Union

import torch
from torch import device
import pdb
import onnxruntime as ort
import numpy as np
import os
import time

from PIL import Image

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import convert_numpy, convert_tensor, to_cuda
from hat.utils.package_helper import require_packages
from hat.models.ir_modules.ir_module import IrModule
from hat.models.ir_modules.utils import np2torch_dtype_dict

try:
    from hbdk4.compiler import load
    from horizon_plugin_pytorch.quantization.hbdk4 import (
        get_hbir_input_flattener,
        get_hbir_output_unflattener,
    )
except ImportError:
    load = None
    get_hbir_input_flattener = None
    get_hbir_output_unflattener = None


__all__ = ["HbirModule"]

logger = logging.getLogger(__name__)


def load_npy_files_in_order(folder_path):
    """
    按文件名数字顺序读取目录下所有 .npy 文件，返回 numpy array 列表
    :param folder_path: 存放 .npy 文件的目录路径
    :return: 按顺序排列的 numpy array 列表
    """
    # 1. 获取目录下所有文件
    file_list = os.listdir(folder_path)

    # 2. 筛选出 .npy 文件，并提取文件名中的数字用于排序
    npy_files = []
    for filename in file_list:
        if filename.endswith(".npy"):
            # 提取文件名中的数字部分（如 "12.npy" → 12）
            num = int(os.path.splitext(filename)[0])
            npy_files.append((num, filename))

    # 3. 按数字从小到大排序
    npy_files.sort(key=lambda x: x[0])

    # 4. 按顺序读取所有 .npy 文件，存入列表
    data_list = []
    for num, filename in npy_files:
        file_path = os.path.join(folder_path, filename)
        filename = file_path
        arr = np.load(file_path)
        # arr = np.transpose(arr, (0, 3, 1, 2))
        data_list.append(arr)
        print(f"已加载: {filename} (shape: {arr.shape})")

    return data_list

@OBJECT_REGISTRY.register
class HbirModuleFisheye(IrModule):
    """Inference module of hbir.

    Args:
         model_path: Path of ir model file.
         return_tensor: Whether to return torch tensor.
         reformat_input_func: Callable function to reformat model inputs.
         reformat_output_func: Callable function to reformat model output.
    """

    @require_packages("horizon_plugin_pytorch>=1.10.3", "hbdk4")
    def __init__(
        self,
        model_path: str,
        return_tensor: bool = True,
        reformat_input_func: Optional[Callable] = None,
        reformat_output_func: Optional[Callable] = None,
    ):
        super().__init__(
            model_path=model_path,
            reformat_input_func=reformat_input_func,
            reformat_output_func=reformat_output_func,
        )
        self.model = load(model_path)
        self.return_tensor = return_tensor
        self.device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )

        self.input_flattener = get_hbir_input_flattener(self.model)
        self.output_unflattener = get_hbir_output_unflattener(self.model)


    def check_input_impl(self, data):
        assert isinstance(data, Dict)
        format_data = {}
        for inp in self.model[0].inputs:
            dtype = np2torch_dtype_dict[inp.type.np_dtype]
            self.check_type_shape(inp.name, inp.type.shape, dtype, data)
            format_data[inp.name] = data[inp.name]
            self.device = data[inp.name].device
        return convert_numpy(format_data)

    def check_output_impl(self, data):
        return_data = data

        if self.return_tensor:
            return_data = convert_tensor(return_data)

        if "cuda" in self.device.type:
            return_data = to_cuda(return_data)
        return return_data

    def forward_impl(self, data):
        # output = self.model.functions[0](*self.input_flattener(data))

        data = {}
        data['img'] = np.load("/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ/10/images.npy")
        data['points0'] = np.load("/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ/10/points0.npy")
        data["points1"] = np.load("/home/Desktop/fisheyedod/horizon/demo/quantizedPTQ/10/points1.npy")

        ort_session = ort.InferenceSession("/home/Desktop/fisheyedod/horizon/ckptsave/float.onnx")
        outputs_info = ort_session.get_outputs()
        inputs_info = ort_session.get_inputs()

        flashocc_input0 = data["img"]
        flashocc_input1 = data["points0"]
        flashocc_input2 = data["points1"]

        output = ort_session.run(None, {inputs_info[0].name: flashocc_input0, inputs_info[1].name: flashocc_input1, inputs_info[2].name: flashocc_input2 })

        kk = 0

        # ort_session = ort.InferenceSession("/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tmp_models/fcos3d_efficientnetb0_nuscenes/fcos3d_b_nhwc.onnx")
        # ort_session = ort.InferenceSession("/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tmp_models/fcos3d_efficientnetb0_nuscenes/float.onnx")

        # outputs_info = ort_session.get_outputs()
        # inputs_info = ort_session.get_inputs()

        # input_img = data["img"]

        # # input_img2 = np.load("/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/debug/transformers_img1773897940.5341728.npy")

        # pdb.set_trace()

        # timestamp =  time.time()
        # file_name0 = "img.npy"
        # file_name1 = "points0.npy"
        # file_name2 = "points1.npy"

        # path_dir = "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/shell/occ/debug"

        # os.makedirs(os.path.join(path_dir, str(timestamp)), exist_ok=True)

        # np.save(os.path.join(path_dir, str(timestamp),  file_name0), flashocc_input0)
        # np.save(os.path.join(path_dir, str(timestamp),  file_name1), flashocc_input1)
        # np.save(os.path.join(path_dir, str(timestamp),  file_name2), flashocc_input2)



        # # npy_data = np.squeeze(input_img, axis=0)  # 指定删除第0个维度
        # # # 1. 转换通道顺序：C×H×W → H×W×C
        # # arr_transposed = npy_data.transpose(1, 2, 0)
        # # # 2. 缩放至 0-255 并转为 uint8 类型
        # # arr_uint8 = (arr_transposed * 255).astype(np.uint8)

        # # # 3. 转换为图像并保存
        # # img = Image.fromarray(arr_uint8)
        # # img.save("/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/demo/fcos3d_efficientnetb0_nuscenes/1533151269512404/fcos3d_input.png")  # 保存为png，也可改为"output.jpg"等格式

        # output2 = ort_session.run(None, {inputs_info[0].name: input_img})


        # np.save("/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/compare_npy/input_ori.npy", input_img)

        # output = load_npy_files_in_order("/home/Desktop/fisheyedod/horizon/infer_out_/quanted")
        # output = load_npy_files_in_order("/home/Desktop/fisheyedod/horizon/infer_out_/ori")
        # output = load_npy_files_in_order("/home/Desktop/fisheyedod/horizon/infer_out_/board_output_npy")
        # output4 = load_npy_files_in_order("/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/compare_npy/quanted")
        # output5 = load_npy_files_in_order("/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/board_output_npy")
        # output6 = load_npy_files_in_order("/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/board_cpp_npy")
        # output7 = load_npy_files_in_order("/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/j6_board_npy")
        # # pdb.set_trace()

        # output2 = [arr.transpose(0, 3, 1, 2) for arr in output6]

        output = self.output_unflattener(output)
        return output



    def cpu(self):
        return self

    def cuda(self, device: Optional[Union[int, device]] = None):
        logger.warnings("Hbir does not support GPU now. Use CPU instead.")
        return self