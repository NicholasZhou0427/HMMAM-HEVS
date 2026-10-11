from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Optional


@dataclass
class VRPOption:
    # ID
    OptionID: str
    DataTime: str

    # Problem
    problem: Literal['cvrp','evrp','sdevrp','evrptw','hcevrp','hcevrpstw'] = 'evrp'
    hc_obj: Literal['min-max','min-sum'] = 'min-sum'
    hc_VEHICLE_CAPACITY:List = field(
        default_factory=lambda: [10, 10, 10]
    )
    hc_VEHICLE_Q:List = field(
        default_factory=lambda: [200,200]
    )
    hc_VEHICLE_v: List = field(
        default_factory=lambda: [200, 200]
    )
    hc_VEHICLE_g: List = field(
        default_factory=lambda: [200, 200]
    )
    hc_VEHICLE_r: List = field(
        default_factory=lambda: [200, 200]
    )
    VEHICLE_Q: float = 2.39
    Refuel_rate: float = 0.13
    STW_penalty_start:float = 1
    STW_penalty_end: float = 1
    charging_penalty_coeff: float = 0.5  # 充电桩惩罚系数：penalty = coeff * (剩余电量 / 总电量)
    unfinished_penalty_coeff: float = 100.0  # 未完成惩罚系数：penalty = coeff * (未访问客户数 + 未返回车辆数)，需与路径cost同量级
    cs_suppress_threshold: float = 0.5   # 充电桩屏蔽阈值：可达客户比例超过此值时屏蔽充电桩
    cs_suppress_min_unvisited: int = 5    # 充电桩屏蔽最少未访问客户数，低于此值不触发屏蔽

    # Data
    train_data_path: Optional[List] = None
    val_data_path: Optional[List] = None
    customer_size: int = 20
    batch_size: int = 14
    train_epoch_size: int = 600
    num_workers: int = 0
    val_epoch_size: int = 30
    model_low_path: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Model
    is_HR_low: bool = False
    is_HR_high: bool = False
    embedding_dim: int = 256
    FF_dim: int = 512
    hc_encoder_FF_dim: int = 512
    n_encode_layers: int = 6
    n_heads: int = 8
    hc_decoder_FF_dim: int = 512
    tanh_clipping: float = 10
    fusion_hidden_dim: int = 256  # MoE gate网络的隐藏层维度
    moe_gate_init_bias: float = 2.0  # MoE gate输出层bias初始化值，正值给高层expert、负值给低层expert，确保训练初期高层主导（softmax([b,-b])）
    # 解码（选车 / 选点）使用 decode_type="topK" 时的候选数；
    # 选点取 min(TopK, graph_size)，选车取 min(TopK, VEHICLE_num)
    TopK: int = 5
    # 训练阶段（REINFORCE rollout 采样）使用的 decode_type：
    #   - "sampling": 标准 multinomial 采样（默认）
    #   - "topK":     在 TopK 候选内做 multinomial 采样，缓解长尾噪声
    # 注意：训练阶段不应使用 "greedy"，否则采样路径退化为确定性
    # 与 baseline rollout 同分布，advantage 恒为 0、无法学习
    train_decode_type: Literal['sampling', 'topK'] = 'sampling'
    dropout_prob: float = 0

    # Training
    lr_model: float = 1e-4
    lr_fusion: float = 1e-4  # 融合MLP的独立学习率
    lr_fusion_decay: float = 0.5  # 融合MLP学习率衰减系数（比主模型更激进）
    weight_decay: float = 1e-2
    lr_decay: float = 0.7
    test_only: bool = False
    n_epochs: int = 10

    #baseline
    baseline: Literal['rollout'] = 'rollout'
    # Baseline 计算模式：
    #   'scst' : Self-Critical Sequence Training —— 当前模型 sampling 采样作为 baseline，
    #            无需冻结快照、无需 t 检验，简单直接。
    #   'pomo' : POMO —— 冻结快照模型 + 多采样 + t 检验控制更新
    baseline_mode: Literal['scst', 'pomo'] = 'pomo'
    bl_alpha: float = 0.05
    # 弱化版 POMO：对同一实例多次独立采样取均值作 baseline，降低 advantage 方差
    # 设为 1 则退化为原始 rollout baseline
    pomo_samples: int = 5
    # POMO 采样模式：
    #   'sequential' - 顺序多次 forward（默认），CPU 友好，显存占用低
    #   'batch'      - 将 batch 数据沿 batch 维复制 pomo_samples 倍后单次 forward，
    #                  充分利用 GPU 并行，速度更快，但显存占用为 sequential 的 pomo_samples 倍
    pomo_mode: Literal['sequential', 'batch'] = 'sequential'
    # 熵正则系数：鼓励策略保持探索，防止收敛后期策略退化
    # 建议从 0.01 开始，过大会阻碍收敛
    entropy_coeff: float = 0.01
    # EMA baseline：每 batch 后用 EMA 渐进更新 baseline 模型参数，降低 baseline 跳变
    # 启用后 POMO 采样会使用 EMA baseline 模型而非当前训练模型
    use_ema_baseline: bool = False
    baseline_ema_alpha: float = 0.05  # EMA 滤波系数，越大跟踪越快（建议 0.01~0.1）
    # 自适应 baseline 模式切换（POMO ↔ SCST）：根据 epoch 数量决定切换时机
    # 前 adaptive_bl_switch_epoch 个 epoch：POMO（冻结快照 + 多采样）
    # 之后切换到 SCST（当前模型 + 单采样）
    adaptive_bl_decode: bool = False
    adaptive_bl_switch_epoch: int = 5      # 从第几个 epoch 开始切换到 SCST（0-based，即前5个epoch用POMO）
    adaptive_bl_min_samples: int = 1       # 稳定期（SCST）使用的采样数


    # Others
    seed: int = 3407
    max_grad_norm: float = 2.0

    # Save
    save_dir:str = './output/output_c8'
