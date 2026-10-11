"""仅使用已发布 .pt 数据的训练、评估和冒烟入口。"""
import argparse
import json
import os
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from WandbOption.MyOption import VRPOption
from Trans_new import (Trans_Model_Encoder, Trans_Model_Decoder_low,
                       Trans_Model_Decoder_high, Trans_Model_low, Trans_Model_high)
from baseline_new import RolloutBaseline_new
from functions import generate_data, load_pt, prepare_folder
from train import evaluate, my_train

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = ROOT / 'data_high/c20v3_20260601_141156_train.pt'
DEFAULT_VAL = ROOT / 'data_high/c20v3_20260601_141156_val.pt'
DEFAULT_LOW = ROOT / 'models/low_c20.pt'


def configure_options(opts, device):
    opts.device = torch.device(device)
    opts.graph_size = opts.customer_size + 21
    opts.VEHICLE_num = len(opts.hc_VEHICLE_CAPACITY)
    return opts


def build_low(opts, device):
    configure_options(opts, device)
    return Trans_Model_low(
        Trans_Model_Encoder(opts.embedding_dim, opts.n_encode_layers, opts.n_heads, opts.FF_dim, opts),
        Trans_Model_Decoder_low(opts.embedding_dim, opts), opts,
    ).to(device)


def build_model(opts, device, low_checkpoint=None, low_options=None):
    configure_options(opts, device)
    if not opts.is_HR_high:
        return build_low(opts, device)
    payload = load_pt(low_checkpoint, map_location=device) if low_checkpoint else None
    if low_options is not None:
        low_opts = VRPOption(**low_options)
    elif payload is not None:
        low_opts = VRPOption(**payload['options'])
    else:
        low_opts = replace(opts, is_HR_low=True, is_HR_high=False)
    if low_opts.customer_size != opts.customer_size:
        raise ValueError('低层权重的客户数必须与当前数据一致')
    low_opts.hc_VEHICLE_CAPACITY = list(opts.hc_VEHICLE_CAPACITY)
    for field in ['VEHICLE_Q', 'Refuel_rate', 'STW_penalty_start', 'STW_penalty_end',
                  'charging_penalty_coeff', 'unfinished_penalty_coeff',
                  'cs_suppress_threshold', 'cs_suppress_min_unvisited']:
        setattr(low_opts, field, getattr(opts, field))
    low = build_low(low_opts, device)
    if payload is not None:
        low.load_state_dict(payload['model'], strict=True)
    return Trans_Model_high(
        Trans_Model_Encoder(opts.embedding_dim, opts.n_encode_layers, opts.n_heads, opts.FF_dim, opts),
        Trans_Model_Decoder_high(opts.embedding_dim, opts), opts, low,
    ).to(device)


def save_checkpoint(path, model, opts, optimizer=None, epoch=0):
    inner = model.module if hasattr(model, 'module') else model
    payload = {'schema_version': 1, 'model': inner.state_dict(), 'options': asdict(opts), 'epoch': epoch}
    if opts.is_HR_high:
        payload['low_options'] = asdict(inner.model_low.opts)
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_model_checkpoint(path, device):
    payload = load_pt(path, map_location=device)
    opts = VRPOption(**payload['options'])
    model = build_model(opts, device, low_options=payload.get('low_options'))
    model.load_state_dict(payload['model'], strict=True)
    return model


def run(opts):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    device = torch.device(opts.device)
    if distributed:
        if device.type != 'cuda':
            raise ValueError('多卡 DDP 需要 CUDA；CPU 请使用单进程')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group('nccl')
        rank = dist.get_rank()
    else:
        rank = 0
    try:
        torch.manual_seed(opts.seed + rank)
        if opts.checkpoint_path:
            model = load_model_checkpoint(opts.checkpoint_path, device)
            loaded_opts = model.opts
            for name in ['train_data_path', 'val_data_path', 'batch_size', 'num_workers', 'n_epochs',
                         'save_dir', 'test_only', 'lr_model', 'lr_fusion', 'lr_decay', 'lr_fusion_decay',
                         'baseline_mode', 'pomo_samples', 'pomo_mode', 'train_decode_type']:
                setattr(loaded_opts, name, getattr(opts, name))
            for name in ['max_train_batches', 'max_val_batches', 'max_train_samples', 'max_val_samples']:
                if hasattr(opts, name):
                    setattr(loaded_opts, name, getattr(opts, name))
            opts = loaded_opts
        else:
            model = build_model(opts, device, low_checkpoint=opts.model_low_path if opts.is_HR_high else None)
        configure_options(opts, device)
        opts.rank = rank
        if opts.test_only:
            dataset = generate_data(opts, is_val=True)
            loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False)
            value = evaluate(model, loader, device, getattr(opts, 'max_val_batches', None))
            return {'validation_cost': value}
        baseline = RolloutBaseline_new(model, opts)
        gate_ids = {id(p) for name, p in model.named_parameters() if name.startswith('gate_')}
        ordinary = [p for p in model.parameters() if p.requires_grad and id(p) not in gate_ids]
        gates = [p for p in model.parameters() if p.requires_grad and id(p) in gate_ids]
        groups = [{'params': ordinary, 'lr': opts.lr_model}]
        if gates:
            groups.append({'params': gates, 'lr': opts.lr_fusion, 'weight_decay': 0.0})
        optimizer = torch.optim.AdamW(groups, weight_decay=opts.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [lambda epoch: opts.lr_decay ** epoch]
            + ([lambda epoch: opts.lr_fusion_decay ** epoch] if gates else []))
        low = model.model_low if opts.is_HR_high else None
        if distributed:
            model = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
        return my_train(model, low, baseline, optimizer, scheduler, opts, device)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['low', 'high'], default='high')
    parser.add_argument('--train-data', nargs='+', default=[str(DEFAULT_TRAIN)])
    parser.add_argument('--val-data', nargs='+', default=[str(DEFAULT_VAL)])
    parser.add_argument('--low-checkpoint', default=str(DEFAULT_LOW))
    parser.add_argument('--checkpoint', default=None, help='加载已保存的模型权重')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--pomo-samples', type=int, default=20)
    parser.add_argument('--baseline-mode', choices=['scst', 'pomo'], default='scst')
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=None)
    parser.add_argument('--output', default=str(ROOT / 'output/train'))
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='读取少量已有验证样本，完成前后向和权重重载')
    args = parser.parse_args()
    for name in ['epochs', 'batch_size', 'embedding_dim', 'layers', 'heads', 'pomo_samples',
                 'max_train_batches', 'max_val_batches']:
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(name + ' 必须大于零')
    if args.smoke and (args.test_only or args.checkpoint):
        parser.error('--smoke 是完整训练冒烟，不能与 --test-only / --checkpoint 合用')
    if args.test_only and not args.checkpoint:
        parser.error('--test-only 必须指定 --checkpoint')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('当前 PyTorch 环境没有可用 CUDA')
    if args.smoke:
        args.epochs, args.batch_size = 1, 2
        args.embedding_dim, args.layers, args.heads = 32, 1, 4
        args.pomo_samples = 2
        args.max_train_batches, args.max_val_batches = 2, 1
        args.train_data = args.val_data
    if args.embedding_dim % args.heads:
        parser.error('embedding-dim 必须能被 heads 整除')
    torch.set_num_threads(1 if args.smoke else max(1, min(torch.get_num_threads(), 8)))
    opts = VRPOption(
        OptionID='public', DataTime=datetime.now().strftime('%Y%m%d_%H%M%S'), problem='hcevrpstw',
        customer_size=20, hc_VEHICLE_CAPACITY=[10, 10, 10],
        STW_penalty_start=0.1, STW_penalty_end=0.1,
        train_data_path=args.train_data, val_data_path=args.val_data,
        model_low_path=args.low_checkpoint, checkpoint_path=args.checkpoint,
        is_HR_low=args.mode == 'low', is_HR_high=args.mode == 'high',
        embedding_dim=args.embedding_dim, n_encode_layers=args.layers, n_heads=args.heads,
        FF_dim=64 if args.smoke else 512, hc_decoder_FF_dim=64 if args.smoke else 512,
        fusion_hidden_dim=16 if args.smoke else 64, moe_gate_init_bias=0.0,
        batch_size=args.batch_size, n_epochs=args.epochs, num_workers=0,
        baseline_mode=args.baseline_mode, pomo_mode='batch', pomo_samples=args.pomo_samples,
        lr_model=1e-4, lr_fusion=1e-3, lr_decay=0.9, lr_fusion_decay=0.7,
        save_dir=args.output, test_only=args.test_only,
    )
    opts.device = torch.device(device)
    opts.max_train_batches = args.max_train_batches
    opts.max_val_batches = args.max_val_batches
    if args.smoke:
        opts.max_train_samples, opts.max_val_samples = 4, 2
    report = run(opts)
    if args.smoke:
        restored = load_model_checkpoint(report['checkpoint'], torch.device(device))
        configure_options(opts, device)
        loader = DataLoader(generate_data(opts, is_val=True), batch_size=2, shuffle=False)
        reloaded_cost = evaluate(restored, loader, torch.device(device), max_batches=1)
        if abs(reloaded_cost - report['best_validation_cost']) > 1e-4:
            raise AssertionError('重载后的评估结果不一致')
        report['reload_validation_cost'] = reloaded_cost
        report['smoke_passed'] = True
    if int(os.environ.get('RANK', '0')) == 0:
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    cli()
