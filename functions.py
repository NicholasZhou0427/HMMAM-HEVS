import inspect
from pathlib import Path

import torch


def load_pt(path, map_location='cpu'):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    kwargs = {'map_location': map_location}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = True
    return torch.load(path, **kwargs)


def move_to(value, device):
    if isinstance(value, dict):
        return {key: move_to(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to(item, device) for item in value)
    return value.to(device) if isinstance(value, torch.Tensor) else value


def generate_data(opts, is_val):
    """兼容原调用名称，仅加载已有文件。"""
    from data.Hevrpstw_data import HEVRPSTWDatasetFromFile
    key = 'val_data_path' if is_val else 'train_data_path'
    paths = getattr(opts, key, None)
    if not paths:
        raise ValueError(f'必须指定 {key}，公开版仅支持已有 .pt 数据')
    limit = getattr(opts, 'max_val_samples' if is_val else 'max_train_samples', None)
    return HEVRPSTWDatasetFromFile(paths, max_samples=limit, customer_size=opts.customer_size)


def prepare_folder(target_path):
    Path(target_path).mkdir(parents=True, exist_ok=True)
