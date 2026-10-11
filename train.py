import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from functions import generate_data, move_to, prepare_folder

accumulation_steps = 2


class AdaptiveDecodeTracker:
    """自适应 baseline 模式切换器：POMO ↔ SCST。

    根据 epoch 数量决定切换时机：
      - epoch < switch_epoch → POMO 模式（冻结快照多采样，n=base_samples）
      - epoch >= switch_epoch → SCST 模式（当前模型单采样，n=min_samples）
    """

    def __init__(self,
                 switch_epoch: int = 5,
                 base_samples: int = 20,
                 min_samples: int = 1):
        self.switch_epoch = switch_epoch
        self.base_samples = base_samples
        self.min_samples = min_samples
        self.current_mode = 'pomo'     # 初始为 POMO 模式
        self.current_samples = base_samples  # 当前生效的采样数

    def update(self, epoch: int):
        """根据当前 epoch 决定是否切换模式；
        返回 (switch_msg | None, current_mode, current_samples)。
        """
        target_mode = 'scst' if epoch >= self.switch_epoch else 'pomo'
        target_samples = self.min_samples if epoch >= self.switch_epoch else self.base_samples

        if target_mode != self.current_mode or target_samples != self.current_samples:
            prev_mode = self.current_mode
            prev_samples = self.current_samples
            self.current_mode = target_mode
            self.current_samples = target_samples
            msg = (
                "[AdaptiveBL] {} (n={}) → {} (n={})  "
                "(epoch={}, switch_epoch={})"
            ).format(prev_mode, prev_samples, target_mode, target_samples,
                     epoch, self.switch_epoch)
            return msg, self.current_mode, self.current_samples
        return None, self.current_mode, self.current_samples

    @property
    def is_refined(self) -> bool:
        return self.current_mode == 'scst'

def train_batch(model, baseline, optimizer, opts, batch_data, batch_id, device,
                bl_decode_type: str = None, bl_n_samples: int = None,
                baseline_mode: str = None):
    batch_data = move_to(batch_data, device)
    cost, log_likelihood, ll_veh, _, cost_info = model(batch_data)

    # ---- Baseline 计算 ----
    # 两种模式：
    #   'scst' : Self-Critical Sequence Training —— 当前模型 sampling 采样作为 baseline，无需 t 检验
    #   'pomo' : POMO —— 冻结快照模型 + 多采样 + t 检验控制更新
    baseline_mode = baseline_mode if baseline_mode is not None else getattr(opts, 'baseline_mode', 'pomo')
    bl_step_costs = None  # 低层模型无 step_costs，保持 None 以触发 fallback

    if baseline_mode == 'scst':
        # ---- SCST: 当前模型 sampling 多次采样取均值作为 baseline ----
        n_samples = bl_n_samples if bl_n_samples is not None else getattr(opts, 'pomo_samples', 4)
        pomo_mode = getattr(opts, 'pomo_mode', 'sequential')

        # SCST 采样 baseline 时用 inner model（绕过 DDP），避免 DDP 重复标记参数 ready
        _inner = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
            if n_samples <= 1:
                bl_cost, _, _, _, bl_cost_info = _inner(batch_data)
                bl_step_costs = bl_cost_info.get('step_costs', None)
            elif pomo_mode == 'batch':
                tiled = _tile_batch_data(batch_data, n_samples)
                c_all, _, _, _, ci_all = _inner(tiled)
                bl_cost = c_all.reshape(n_samples, *cost.shape).mean(dim=0)
                if 'step_costs' in ci_all:
                    bl_step_costs = ci_all['step_costs'].reshape(
                        n_samples, cost.shape[0], -1).mean(dim=0)
            else:
                sample_costs = []
                sample_step_costs = []
                for _ in range(n_samples):
                    c_i, _, _, _, ci_info = _inner(batch_data)
                    sample_costs.append(c_i)
                    if 'step_costs' in ci_info:
                        sample_step_costs.append(ci_info['step_costs'])
                bl_cost = torch.stack(sample_costs, dim=0).mean(dim=0)
                if sample_step_costs:
                    bl_step_costs = torch.stack(sample_step_costs, dim=0).mean(dim=0)

    else:
        # ---- POMO: 冻结快照模型 + 多采样 + t 检验控制更新 ----
        n_samples = bl_n_samples if bl_n_samples is not None else getattr(opts, 'pomo_samples', 4)
        pomo_mode = getattr(opts, 'pomo_mode', 'sequential')
        _bl_dt = bl_decode_type if bl_decode_type is not None else getattr(opts, 'train_decode_type', 'sampling')
        if n_samples > 1:
            pomo_model = baseline.model
            bl_inner = pomo_model.module if hasattr(pomo_model, 'module') else pomo_model
            bl_inner.decoder.set_decode_type(_bl_dt)
            pomo_model.train()
            if pomo_mode == 'batch':
                with torch.no_grad():
                    tiled = _tile_batch_data(batch_data, n_samples)
                    c_all, _, _, _, ci_all = pomo_model(tiled)
                    bl_cost = c_all.reshape(n_samples, *cost.shape).mean(dim=0)
                    if 'step_costs' in ci_all:
                        bl_step_costs = ci_all['step_costs'].reshape(
                            n_samples, cost.shape[0], -1).mean(dim=0)
            else:
                with torch.no_grad():
                    sample_costs = []
                    sample_step_costs = []
                    for _ in range(n_samples):
                        c_i, _, _, _, ci_info = pomo_model(batch_data)
                        sample_costs.append(c_i)
                        if 'step_costs' in ci_info:
                            sample_step_costs.append(ci_info['step_costs'])
                    bl_cost = torch.stack(sample_costs, dim=0).mean(dim=0)
                    if sample_step_costs:
                        bl_step_costs = torch.stack(sample_step_costs, dim=0).mean(dim=0)
            bl_inner.decoder.set_decode_type('greedy')
            pomo_model.eval()
        else:
            pomo_model = baseline.model
            bl_inner = pomo_model.module if hasattr(pomo_model, 'module') else pomo_model
            bl_inner.decoder.set_decode_type(_bl_dt)
            pomo_model.train()
            with torch.no_grad():
                bl_cost, _, _, _, bl_cost_info = pomo_model(batch_data)
                bl_step_costs = bl_cost_info.get('step_costs', None)
            bl_inner.decoder.set_decode_type('greedy')
            pomo_model.eval()

    cost = cost.view(-1)
    bl_cost = bl_cost.view(-1)
    log_likelihood = log_likelihood.view(-1)
    ll_veh = ll_veh.view(-1)

    assert cost.shape == bl_cost.shape == log_likelihood.shape == ll_veh.shape

    advantage_raw = (cost - bl_cost).detach()

    # --- Reward-to-Go loss（仅高层模型有 step_costs；低层模型 fallback 到 trajectory-level）---
    if 'step_costs' in cost_info and 'log_prob_per_step' in cost_info and bl_step_costs is not None:
        # 模型的 reward-to-go: R_t = sum_{k=t}^{T} step_cost_k
        model_step_costs = cost_info['step_costs']  # [batch, T]
        model_rtg = torch.flip(
            torch.cumsum(torch.flip(model_step_costs, dims=[1]), dim=1),
            dims=[1]
        )  # [batch, T]
        # Baseline 的 reward-to-go
        bl_rtg = torch.flip(
            torch.cumsum(torch.flip(bl_step_costs, dims=[1]), dim=1),
            dims=[1]
        )  # [batch, T]
        # Per-step advantage: 模型 R_t - baseline R_t
        advantage_per_step = (model_rtg - bl_rtg).detach()  # [batch, T]

        # Per-step log_prob（保留梯度）
        log_prob_per_step = cost_info['log_prob_per_step']  # [batch, T]

        # 只对有效步计算 loss（无效步 log_prob=0，不贡献梯度）
        reinforce_loss = torch.mean(
            (advantage_per_step * log_prob_per_step).sum(dim=1)
        )
    else:
        # Fallback: 低层模型使用原版 trajectory-level REINFORCE
        log_prob = log_likelihood + ll_veh
        reinforce_loss = torch.mean(advantage_raw * log_prob)

    # 信息熵估计（仅用于监控）
    log_prob = log_likelihood + ll_veh
    entropy = -log_prob.detach().mean()

    reinforce_loss = reinforce_loss / accumulation_steps
    reinforce_loss.backward()
    # loss_ys += reinforce_loss

    # 量纲说明：reinforce_loss 已除以 accumulation_steps 后再 backward，所以 accumulation_steps 次累加完成后，
    # param.grad == （1/accumulation_steps） * Σ_i d(L_i)/d(param) == 「大 batch 平均梯度」。
    # 默认不采集：只在最后一次累加完（optimizer.step()/zero_grad() 之前）才采集一次，
    # 避免奇偶 batch 梯度累加次数不同造成的统计口径交替，同时保证 total_norm 与逐层 norm 都是
    # 「大 batch 平均梯度」的 L2 范数，量纲一致。
    layer_norms, total_norm = [], None
    if (batch_id + 1) % accumulation_steps == 0:
        # ---- 逐层 gradient norm 采集（最后一次累加完、step 与 zero_grad 之前）----
        layer_norms, total_norm = _collect_grad_norms(model, p_ord=2.0)
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    return cost, bl_cost, advantage_raw, advantage_raw, layer_norms, total_norm, entropy, cost_info

def _tile_batch_data(batch_data, n: int):
    """沿 batch 维度将 batch_data 重复 n 次，用于 GPU 并行 POMO 采样。

    支持以下递归结构：
      - torch.Tensor : [B, ...] -> [B*n, ...]
      - dict         : 递归对每个 value 处理
      - list / tuple : 递归对每个元素处理，保留原始容器类型
      - 其他（int / float / None 等叶节点）: 原样返回
    """
    if isinstance(batch_data, torch.Tensor):
        repeat_dims = (n,) + (1,) * (batch_data.dim() - 1)
        return batch_data.repeat(*repeat_dims)
    elif isinstance(batch_data, dict):
        return {k: _tile_batch_data(v, n) for k, v in batch_data.items()}
    elif isinstance(batch_data, (list, tuple)):
        tiled = [_tile_batch_data(x, n) for x in batch_data]
        return type(batch_data)(tiled)
    else:
        return batch_data

def _collect_grad_norms(model, p_ord=2.0):
    """遍历 model.named_parameters()，返回 [(name, ||grad||_p), ...] 与 total_norm。

    - 只统计 requires_grad=True 且 grad 已填充（backward 后）的参数
    - total_norm 与 torch.nn.utils.clip_grad_norm_ 一致：sqrt(sum(||g||_2^2))
    - 调用者应保证在「累加完成」的那一刻调用，此时 param.grad 是大 batch 的平均梯度，
      所以返回的逐层 norm 与 total_norm 都是「大 batch 平均梯度」的 L2 范数，量纲一致。
    """
    layer_norms = []
    total_sq = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        gn = param.grad.detach().data.norm(p_ord).item()
        layer_norms.append((name, gn))
        total_sq += gn * gn
    total_norm = total_sq ** 0.5
    return layer_norms, total_norm

def evaluate(model, dataloader, device, max_batches=None):
    inner = model.module if hasattr(model, 'module') else model
    inner.eval()
    inner.decoder.set_decode_type('greedy')
    total, count = 0.0, 0
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            if max_batches is not None and batch_id >= max_batches:
                break
            cost = inner(move_to(batch, device))[0]
            if not torch.isfinite(cost).all():
                raise FloatingPointError('验证成本包含非有限值')
            total += cost.sum().item()
            count += cost.numel()
    if count == 0:
        raise ValueError('验证集不能为空')
    return total / count


def my_train(model, model_low, baseline, optimizer, lr_scheduler, opts, device):
    from main import save_checkpoint
    prepare_folder(opts.save_dir)
    training_dataset = generate_data(opts, is_val=False)
    val_dataset = generate_data(opts, is_val=True)
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    sampler = DistributedSampler(training_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(training_dataset, batch_size=opts.batch_size,
                              sampler=sampler, shuffle=sampler is None,
                              num_workers=opts.num_workers)
    # rank 0 验证完整数据，不对验证集做分片。
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size,
                            shuffle=False, num_workers=opts.num_workers)
    inner = model.module if hasattr(model, 'module') else model
    if opts.train_decode_type not in ('sampling', 'topK'):
        raise ValueError('训练解码必须采用 sampling 或 topK')
    if opts.baseline_mode not in ('scst', 'pomo'):
        raise ValueError('baseline_mode 必须为 scst 或 pomo')
    tracker = AdaptiveDecodeTracker(
        switch_epoch=opts.adaptive_bl_switch_epoch,
        base_samples=opts.pomo_samples,
        min_samples=opts.adaptive_bl_min_samples,
    ) if opts.adaptive_bl_decode else None
    best = float('inf')
    train_batches, optimizer_steps = 0, 0
    optimizer.zero_grad()
    max_batches = getattr(opts, 'max_train_batches', None)
    for epoch in range(opts.n_epochs):
        baseline_mode, baseline_samples = opts.baseline_mode, opts.pomo_samples
        if tracker is not None:
            message, baseline_mode, baseline_samples = tracker.update(epoch)
            if message and rank == 0:
                print(message)
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        inner.decoder.set_decode_type(opts.train_decode_type)
        if model_low is not None:
            model_low.eval()
        epoch_batches = 0
        for batch_id, batch in enumerate(train_loader):
            if max_batches is not None and batch_id >= max_batches:
                break
            result = train_batch(
                model, baseline, optimizer, opts, batch, batch_id, device,
                bl_decode_type=opts.train_decode_type, bl_n_samples=baseline_samples,
                baseline_mode=baseline_mode,
            )
            if not torch.isfinite(result[0]).all():
                raise FloatingPointError('训练成本包含非有限值')
            epoch_batches += 1
            train_batches += 1
            if (batch_id + 1) % accumulation_steps == 0:
                optimizer_steps += 1
            if getattr(opts, 'use_ema_baseline', False):
                baseline.ema_update(inner, opts.baseline_ema_alpha)
            if baseline_mode == 'pomo':
                baseline.epoch_callback(result[0], result[1], model)
            if rank == 0:
                print(f'epoch={epoch + 1} batch={batch_id + 1} cost={result[0].mean().item():.6f}')
        if epoch_batches == 0:
            raise ValueError('训练集不能为空')
        remainder = epoch_batches % accumulation_steps
        if remainder:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(accumulation_steps / remainder)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_steps += 1
        if rank == 0:
            validation = evaluate(inner, val_loader, device, getattr(opts, 'max_val_batches', None))
            save_checkpoint(str(opts.save_dir) + '/last.pt', inner, opts, optimizer, epoch + 1)
            if validation < best:
                best = validation
                save_checkpoint(str(opts.save_dir) + '/best.pt', inner, opts, optimizer, epoch + 1)
            print(f'validation_cost={validation:.6f}')
        if distributed:
            dist.barrier()
        lr_scheduler.step()
    return {'train_batches': train_batches, 'optimizer_steps': optimizer_steps,
            'best_validation_cost': best if rank == 0 else None,
            'checkpoint': str(opts.save_dir) + '/last.pt'}
