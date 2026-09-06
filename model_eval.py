"""模型轻量化分析：自动统计参数量（Params）、计算量（FLOPs）与单图推理延迟
基于 PyTorch 原生 hook 机制实现，无需安装 thop/fvcore 等第三方库。
结果同时打印到终端并保存至 model_profile.json（供 streamlit_app.py 展示）。
"""
import json
import time
import torch
import torch.nn as nn
import yaml
from core.model import SSD


def count_parameters(model):
    """统计总参数量与可训练参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _conv2d_flops(module, input, output):
    x = input[0]
    out_h, out_w = output.shape[2], output.shape[3]
    kernel_ops = module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
    # 每个输出像素：in_c/g * k * k 次乘加，再乘输出通道数
    flops = out_h * out_w * module.out_channels * kernel_ops
    module.__flops__ += flops


def _linear_flops(module, input, output):
    x = input[0]
    flops = x.shape[:-1].numel() * module.in_features * module.out_features
    module.__flops__ += flops


def _bn_flops(module, input, output):
    # 归一化约 2 次运算/元素，仿射变换再 2 次
    module.__flops__ += input[0].numel() * 2


_FLOPS_HOOKS = {nn.Conv2d: _conv2d_flops, nn.Linear: _linear_flops,
                nn.BatchNorm2d: _bn_flops, nn.GroupNorm: _bn_flops}


def count_flops(model, input_size=(1, 3, 300, 300), device='cpu'):
    """用 forward hook 统计一次前向的 FLOPs（乘加按 1 次计）"""
    model.eval()
    for m in model.modules():
        m.__flops__ = 0
    handles = []
    for m in model.modules():
        hook_fn = _FLOPS_HOOKS.get(type(m))
        if hook_fn is not None:
            handles.append(m.register_forward_hook(hook_fn))
    try:
        with torch.no_grad():
            model(torch.zeros(*input_size).to(device))
        total = sum(m.__flops__ for m in model.modules())
    finally:
        for h in handles:
            h.remove()
        for m in model.modules():
            if hasattr(m, '__flops__'):
                del m.__flops__
    return total


def measure_latency(model, device, input_size=(1, 3, 300, 300), warmup=10, repeats=50):
    """单图推理延迟：先预热再计时，返回平均毫秒数与 FPS"""
    model.eval()
    dummy = torch.zeros(*input_size).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    avg_ms = elapsed / repeats * 1000
    return avg_ms, 1000.0 / avg_ms


def format_size(num):
    """把参数量/FLOPs 格式化为易读字符串"""
    if num >= 1e9:
        return f"{num / 1e9:.2f} G"
    if num >= 1e6:
        return f"{num / 1e6:.2f} M"
    if num >= 1e3:
        return f"{num / 1e3:.2f} K"
    return str(num)


def model_size_mb(model):
    """估算模型权重占用的显存/内存（MB）"""
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    return bytes_ / (1024 * 1024)


def profile_model(weights_path=None):
    """完整轻量化分析，返回结果字典"""
    with open('config/default.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SSD(backbone=cfg['backbone'], num_classes=cfg['num_classes'])
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    total_params, trainable_params = count_parameters(model)
    # FLOPs 与输入尺寸相关，与权重无关，先在 CPU 上统计避免显存占用
    flops = count_flops(model, (1, 3, model.image_size, model.image_size), device='cpu')
    model.to(device)
    avg_ms, fps = measure_latency(model, device)

    result = {
        'device': str(device),
        'backbone': cfg['backbone'],
        'input_size': model.image_size,
        'num_priors': int(model.priors.shape[0]),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': round(model_size_mb(model), 2),
        'flops': flops,
        'latency_ms': round(avg_ms, 2),
        'fps': round(fps, 1),
    }
    return result


def print_report(result):
    print("=" * 50)
    print("模型轻量化分析报告")
    print("=" * 50)
    print(f"  设备:           {result['device']}")
    print(f"  Backbone:       {result['backbone']}")
    print(f"  输入尺寸:       {result['input_size']}x{result['input_size']}")
    print(f"  锚框总数:       {result['num_priors']}")
    print("-" * 50)
    print(f"  参数量 (Params):    {format_size(result['total_params'])} ({result['total_params']:,})")
    print(f"  可训练参数:         {format_size(result['trainable_params'])}")
    print(f"  模型大小:           {result['model_size_mb']:.2f} MB")
    print(f"  计算量 (FLOPs):     {format_size(result['flops'])} ({result['flops']:,})")
    print(f"  单图推理延迟:       {result['latency_ms']:.2f} ms")
    print(f"  推理速度:           {result['fps']:.1f} FPS")
    print("=" * 50)


if __name__ == '__main__':
    import os
    # 默认加载最新权重（仅用于延迟测试，统计结果与权重无关）
    weights = 'runs/ssd_epoch_150.pth' if os.path.exists('runs/ssd_epoch_150.pth') else None
    result = profile_model(weights)
    print_report(result)

    with open('model_profile.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("结果已保存至 model_profile.json")
