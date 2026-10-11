# HMMAMv4 数据与 MoE 模型

本版本提供 20 客户问题的成品数据、冻结的低层模型，以及 MoE 高层模型的训练和评估入口。代码基于 DDP_MoE 版本整理，只读取已有 `.pt` 文件。发布历史独立起始。本次发布保留与随附数据配套的模型和训练设置，尚未对齐论文终稿的全部模型细节与实验配置。

数据文件位于 `data_high/`：训练集包含 100000 个实例，验证集包含 2000 个实例。文件大小和 SHA256 校验值记录在 `data_manifest.json`。各实例为六个张量组成的字典：

| 字段 | 形状 | 含义 |
|---|---|---|
| `loc` | `[41, 2]` | 节点坐标 |
| `demand` | `[41, 1]` | 节点需求 |
| `start_time` | `[41, 1]` | 时间窗起点 |
| `end_time` | `[41, 1]` | 时间窗终点 |
| `service_time` | `[41, 1]` | 服务时间 |
| `mid_time` | `[41, 1]` | 时间窗中点 |

索引 0 为仓库，1 到 20 为充电桩，21 到 40 为客户。时间、距离和容量沿用文件中的模型单位。这里只提供训练集和验证集，没有独立测试集。

下载仓库前安装 Git LFS，并获取成品数据与权重：

```bash
git lfs install
git clone https://github.com/NicholasZhou0427/HMMAM-HEVS.git
cd HMMAM-HEVS
git lfs pull
```

安装依赖并运行冒烟测试：

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python main.py --smoke --device cpu --output output/smoke-high
python main.py --mode low --smoke --device cpu --output output/smoke-low
```

冒烟模式借用验证文件中的 4 个实例做两个训练批次，再评估前 2 个实例，确认梯度更新、权重保存和重载。高层模型使用小规模网络加速，冻结低层模型仍加载发布的完整权重；低层冒烟使用小规模随机初始化网络。这些运行结果用于检查执行流程，不作为模型性能指标。

训练默认大小的高层模型：

```bash
python main.py --mode high --device cuda --output output/high
```

该入口使用三辆容量为 10 的车、SCST 基线和 MoE 融合；可用 `--baseline-mode pomo` 切换基线。低层权重在 `models/low_c20.pt` 中，包含模型参数和结构配置，不包含历史优化器或基线副本。默认训练网络为 256 维嵌入、6 层、8 个注意力头；完整参数定义在 `WandbOption/MyOption.py`。如需调整实验配置，应同时检查数据规模和权重结构是否匹配。

评估保存的模型：

```bash
python main.py --test-only --checkpoint output/high/best.pt --device cpu
```

`--checkpoint` 加载模型权重与网络结构；继续训练时优化器会重新初始化。`run_multituning.py` 是同一命令行入口的兼容名称。CPU 与单 GPU 可以直接调用；多 GPU 使用 `torchrun` 调用同一入口。

缺少数据文件或未指定数据路径时，程序直接报错。训练输出保存在 `output/`，该目录不进入版本控制，运行入口不会清空已有目录。

本版本已验证 CPU 上的数据加载、低层和高层前向、高层反向传播、真实参数更新和权重保存重载。GPU 与多 GPU 训练需要在相应设备上另行验证；冒烟测试不证明完整训练的收敛或最终效果。

本仓库采用 [MIT 许可证](LICENSE)，版权声明与许可正文见 `LICENSE`。
