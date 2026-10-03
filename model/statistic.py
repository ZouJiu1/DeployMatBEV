import torch

def count_params_from_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 兼容两种存储：直接state_dict / 带key "state_dict"
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    total_params = 0
    layer_info = []
    for name, tensor in state_dict.items():
        # 只统计权重张量，跳过optimizer、epoch、loss等无关key
        if isinstance(tensor, torch.Tensor):
            num = tensor.numel()
            total_params += num
            layer_info.append((name, tensor.shape, num))
    
    print("="*60)
    print(f"权重文件路径: {ckpt_path}")
    print(f"总参数量: {total_params:,}")
    print(f"参数量(MB, float32): {total_params * 4 / 1024 / 1024:.2f} MB")
    print("="*60)
    # 打印前20层查看
    for name, shape, num in layer_info[:20]:
        print(f"{name:40} | shape: {str(shape):20} | params: {num:,}")
    if len(layer_info) > 20:
        print(f"... 省略剩余 {len(layer_info)-20} 层")

if __name__ == "__main__":
    ckpt_file = "your_model.pth"  # 替换权重路径
    count_params_from_ckpt(ckpt_file)