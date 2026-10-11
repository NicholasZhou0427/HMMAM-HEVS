import copy

from scipy.stats import ttest_rel
import numpy as np
import torch
from typing import Mapping


class BaselineTTestError(Exception):
    """诊断 RolloutBaseline t 检验异常的专用异常。

    会带上详细的 reason 、n 、mean_diff 、std_diff 、
    has_nan 、has_inf 等字段便于上层定位。
    """

    def __init__(self, reason, **diag):
        self.reason = reason
        self.diag = diag
        msg = "[BaselineTTestError] reason={} | ".format(reason) + \
              " | ".join(f"{k}={v}" for k, v in diag.items())
        super().__init__(msg)


class RolloutBaseline_new:
    """Rollout Baseline 实现，支持两种模式：

    1. SCST (Self-Critical Sequence Training) 模式（默认）：
       - baseline = 当前训练模型 sampling 多次采样取均值
       - 不需要冻结快照模型，不需要 t 检验
       - advantage = cost - mean(sampled_costs)

    2. POMO 模式：
       - baseline = 冻结快照模型的多采样输出
       - 通过配对 t 检验决定何时更新快照
       - advantage = cost - mean(snapshot_sampled_costs)

    通过 opts.baseline_mode 控制使用哪种模式。
    """
    def __init__(self, model, opts):
        super(RolloutBaseline_new, self).__init__()

        self.opts = opts
        self.model = copy.deepcopy(model)
        # deepcopy 后确保副本在正确设备上（创建 baseline 时 model 可能尚未 .to(device)）
        self.model.to(opts.device)
        self._update_model(model)

    def _update_model(self, model):
        # 兼容 DataParallel：训练模型被包装后 state_dict keys 带 "module." 前缀，
        # 而 self.model 未被包装，需从 inner model 获取 state_dict 以保证 key 匹配
        src_model = model.module if hasattr(model, "module") else model
        self.model.load_state_dict(src_model.state_dict())
        # 兼容是否被 DataParallel 包装：包装后通过 .module 访问，未包装则直接访问
        inner = self.model.module if hasattr(self.model, "module") else self.model
        inner.decoder.set_decode_type("greedy")
        self.model.eval()

    def rollout(self, batch_data, model=None, return_cost_info=False):
        if model is None:
            model = self.model
        with torch.no_grad():
            cost, _, _, _, cost_info = model(batch_data)
        if return_cost_info:
            return cost, cost_info
        return cost

    def ema_update(self, model, alpha: float = 0.05):
        """EMA 渐进式更新 baseline 模型参数：
        bl_params = (1 - alpha) * bl_params + alpha * train_params

        每个 batch 后调用，使 baseline 平滑跟踪训练模型，
        避免 t-test 门控导致 baseline 长期滞后，同时防止 baseline
        因单步参数大跳而引起 advantage 剧烈抖动。
        alpha 越大跟踪越快（建议范围 0.01 ~ 0.1）。
        """
        src = model.module if hasattr(model, 'module') else model
        tgt = self.model.module if hasattr(self.model, 'module') else self.model
        with torch.no_grad():
            for p_tgt, p_src in zip(tgt.parameters(), src.parameters()):
                p_tgt.data.mul_(1.0 - alpha).add_(p_src.data * alpha)

    def load_model(self,model_state_dict:Mapping[str,torch.Tensor]):
        self.model.load_state_dict(model_state_dict)
        # 兼容是否被 DataParallel 包装：包装后通过 .module 访问，未包装则直接访问
        inner = self.model.module if hasattr(self.model, "module") else self.model
        inner.decoder.set_decode_type("greedy")
        self.model.eval()

    def epoch_callback(self, cost, bl_cost, model):
        if (cost.mean() - bl_cost.mean()) < 0:
            ## ———— 输入体检 ————
            cost_np = cost.detach().cpu().numpy().astype(np.float64).ravel()
            bl_np = bl_cost.detach().cpu().numpy().astype(np.float64).ravel()
            diff = cost_np - bl_np

            try:
                # ---------------- 输入体检，定位 tt 可能为 NaN / 非负的三类原因 ----------------
                n = diff.size
                has_nan = bool(np.isnan(cost_np).any() or np.isnan(bl_np).any())
                has_inf = bool(np.isinf(cost_np).any() or np.isinf(bl_np).any())
                mean_diff = float(diff.mean()) if n > 0 else float("nan")
                std_diff = float(diff.std(ddof=1)) if n > 1 else float("nan")

                # 原因三：输入本身含 NaN / Inf（梯度爆炸 / 惩罚 inf / 数据坏掉）
                if has_nan or has_inf:
                    raise BaselineTTestError(
                        "input_has_nan_or_inf",
                        n=n,
                        has_nan=has_nan,
                        has_inf=has_inf,
                        cost_min=float(np.nanmin(cost_np)) if n > 0 else None,
                        cost_max=float(np.nanmax(cost_np)) if n > 0 else None,
                        bl_min=float(np.nanmin(bl_np)) if n > 0 else None,
                        bl_max=float(np.nanmax(bl_np)) if n > 0 else None,
                        hint="cost 或 bl_cost 中出现了 NaN/Inf，检查梯度爆炸、未完成惩罚 inf、数据字段是否缺失。",
                    )

                # 原因二：样本量不足（n<2 时 std 自由度为 0，t 为 NaN）
                if n < 2:
                    raise BaselineTTestError(
                        "sample_size_too_small",
                        n=n,
                        hint="样本量 < 2 无法进行配对 t 检验，检查 batch_size 是否太小或 DataParallel 后剩余样本不足。",
                    )

                # 原因一：差值标准差为 0（cost 与 bl_cost 逐位差值是常数）
                if not np.isfinite(std_diff) or std_diff < 1e-12:
                    # 进一步区分“完全相等”与“差值=非零常数”
                    if abs(mean_diff) < 1e-12:
                        sub_reason = "cost_equals_bl_cost"
                        hint = "cost 与 bl_cost 逐位相等，常见于训练初期 baseline 仍为 model 副本。"
                    else:
                        sub_reason = "diff_is_constant_nonzero"
                        hint = "cost - bl_cost 是个非零常数，std=0 会导致 t 为 inf/NaN，检查是否多个样本被同样的惩罚镐到同一值。"
                    raise BaselineTTestError(
                        "zero_diff_std",
                        n=n,
                        mean_diff=mean_diff,
                        std_diff=std_diff,
                        sub_reason=sub_reason,
                        hint=hint,
                    )

                # ---------------- 检验本体 ----------------
                tt, pp = ttest_rel(cost_np, bl_np)
                tt = float(tt)
                pp = float(pp)

                # 安全网：即使上面都走了，tt 仍可能因数值下溢出现 NaN/inf
                if not np.isfinite(tt):
                    raise BaselineTTestError(
                        "t_stat_not_finite",
                        n=n,
                        mean_diff=mean_diff,
                        std_diff=std_diff,
                        tt=tt,
                        pp=pp,
                        hint="体检通过但 ttest_rel 仍返回了 NaN/Inf，下溢或 scipy 版本问题，考虑改用双精度。",
                    )

                # 与前置条件不一致：mean(cost-bl) < 0 但 tt >= 0。数学上不应发生。
                if tt >= 0:
                    raise BaselineTTestError(
                        "t_stat_non_negative",
                        n=n,
                        mean_diff=mean_diff,
                        std_diff=std_diff,
                        tt=tt,
                        pp=pp,
                        hint="mean(cost-bl)<0 但 tt>=0 与数学预期不符，检查数据是否被意外修改过。",
                    )

                # ---------------- 正常分支：单边 p 值下决定是否更新 baseline ----------------
                p_val = pp / 2
                if p_val < self.opts.bl_alpha:
                    print('Update baseline')
                    self._update_model(model)

            except BaselineTTestError as e:
                # 专用诊断异常：打印 reason / diag，并输出原始 cost_np / bl_np 与 NaN/Inf 检查
                self._dump_diag("BaselineTTestError", e, cost_np, bl_np)
            except Exception as e:
                # 其它未预期异常（例如 scipy 内部错误、长度不一致 ValueError 等）
                self._dump_diag("UnexpectedException", e, cost_np, bl_np)

    @staticmethod
    def _dump_diag(tag, err, cost_np, bl_np):
        """统一打印 baseline 异常现场环境：包括 cost_np / bl_np 原始值及 NaN/Inf 检查。"""
        # NaN / Inf 检查
        cost_has_nan = bool(np.isnan(cost_np).any())
        cost_has_inf = bool(np.isinf(cost_np).any())
        bl_has_nan = bool(np.isnan(bl_np).any())
        bl_has_inf = bool(np.isinf(bl_np).any())

        # numpy 打印限制：超过 64 元素只取首尾，避免刷屏
        with np.printoptions(threshold=64, edgeitems=8, suppress=False, precision=6):
            print("=" * 70)
            print(f"[{tag}] baseline.epoch_callback 出现异常")
            print("-" * 70)
            if isinstance(err, BaselineTTestError):
                print("reason :", err.reason)
                for k, v in err.diag.items():
                    print(f"  {k} : {v}")
            else:
                print("exception_type :", type(err).__name__)
                print("message        :", str(err))
            print("-" * 70)
            print("cost_np.shape =", cost_np.shape, ", dtype =", cost_np.dtype)
            print("cost_np       =", cost_np)
            print("  has_nan = {} | has_inf = {}".format(cost_has_nan, cost_has_inf))
            if cost_has_nan or cost_has_inf:
                bad_idx = np.where(~np.isfinite(cost_np))[0]
                print("  bad_idx (cost) =", bad_idx[:32], "..." if bad_idx.size > 32 else "")
            print("bl_np.shape   =", bl_np.shape, ", dtype =", bl_np.dtype)
            print("bl_np         =", bl_np)
            print("  has_nan = {} | has_inf = {}".format(bl_has_nan, bl_has_inf))
            if bl_has_nan or bl_has_inf:
                bad_idx = np.where(~np.isfinite(bl_np))[0]
                print("  bad_idx (bl)  =", bad_idx[:32], "..." if bad_idx.size > 32 else "")
            print("=" * 70)

    def state_dict(self):
        return {
            'model': self.model.state_dict()
        }
