# 同奖励 PPO / SAC / DQN 实验

更新：2026-11-09。

本轮沿用奖励改进后的 PPO 任务：平地、世界 +X 指令 0.10 m/s、36 维观测、8 个关节动作、50 Hz 控制、10 秒回合，以及完整域随机化。所有奖励项、权重和终止条件与 [交接文档](HANDOVER_20261106.md#2-奖励函数设计) 一致，包括 `-0.5 × (1-cos(航向误差))` 奖励率。环境奖励不缩放；SAC 的熵项属于算法优化目标。

## 基线方法与训练设置

| 方法 | 实现 | 动作与主要设置 |
| --- | --- | --- |
| PPO | 复用奖励改进后的 PPO 最终模型 | 连续 8 维；保留检查点内的观测归一化 |
| SAC | 双 Q、tanh 高斯策略、自动温度 | 连续 8 维；初始温度 0.02、目标熵 -8 |
| DQN | 共享状态价值、分支优势、Double-Q 目标 | 每关节 5 档 `[-1,-0.5,0,0.5,1]`；8 个分支共 40 个 Q 输出 |

DQN 使用分支网络和 Double DQN 更新，每个关节有 5 个离散动作，采用均匀经验回放。动作离散化是相对 PPO/SAC 的额外差异；本实现未包含参考论文的优先经验回放等全部设置。实现参考：[SAC](https://arxiv.org/abs/1812.05905)、[Action Branching](https://arxiv.org/abs/1711.08946)。

每种算法使用 42/43/44 三个随机种子，每个随机种子使用 256 个环境、训练 204.8 万步。SAC/DQN 使用 256×256 ReLU 网络、学习率 3e-4、折扣 0.99、目标网络软更新 0.005；回放容量 262144、先收集 65536 条经验，之后每次并行环境交互后更新 2 次、批量 512，共 15488 次更新。DQN 的 ε 在训练的前 102.4 万步内由 1 降至 0.05。新算法直接使用环境物理缩放后的观测，不额外归一化。

相同环境交互步数不代表相同计算量。仅评估最终模型，不按测试效果挑检查点或调参。本轮不包含使用旧奖励的 Dyna-PPO。

## 训练时间

相同 RTX 4090 D GPU、256 个并行环境、每种子 204.8 万步；表中为种子 42/43/44 的均值 ± 样本标准差。

| 方法 | 日志记录的训练阶段（秒） | 平均约（分钟） | 训练程序总时间（秒） |
| --- | --- | --- | --- |
| PPO | 327.00 ± 0.46 | 5.45 | 334.51 ± 1.64 |
| SAC | 207.43 ± 2.12 | 3.46 | 212.27 ± 2.63 |
| DQN | 170.88 ± 0.64 | 2.85 | 175.35 ± 0.68 |

PPO 的训练阶段时间是日志中采样与更新计时之和，SAC/DQN 为整个训练循环的计时，范围不完全相同。程序总时间包含初始化、训练、最终保存和重载验证，不含退出、单独评估和录像。运行负载与诊断开销没有严格控制，因此这里只报告历史耗时，不据此认定算法加速或成功率提高。Dyna-PPO 的历史耗时及完整定义见[简版报告](HANDOVER_20261106.md#训练时间)，逐种子数据见[训练时间记录](../reports/training_times_20261109.json)。

## 模型权重

下表是本轮结果和 Demo 使用的 **9 个最终模型**，均已保留；各列对应训练随机种子。

| 方法 | 种子 42 | 种子 43 | 种子 44 |
| --- | --- | --- | --- |
| PPO | [final.pt](../runs/ppo_heading_20261105/training/heading_seed42/final.pt) | [final.pt](../runs/ppo_heading_20261105/training/heading_seed43/final.pt) | [final.pt](../runs/ppo_heading_20261105/training/heading_seed44/final.pt) |
| SAC | [final.pt](../runs/offpolicy_heading_20261107/training/sac_seed42/final.pt) | [final.pt](../runs/offpolicy_heading_20261107/training/sac_seed43/final.pt) | [final.pt](../runs/offpolicy_heading_20261107/training/sac_seed44/final.pt) |
| DQN | [final.pt](../runs/offpolicy_heading_20261107/training/dqn_seed42/final.pt) | [final.pt](../runs/offpolicy_heading_20261107/training/dqn_seed43/final.pt) | [final.pt](../runs/offpolicy_heading_20261107/training/dqn_seed44/final.pt) |

路径与哈希见[评估配置](../runs/offpolicy_heading_20261107/evaluation_protocol.json)。包括 Dyna-PPO 在内的全部保留权重见[简版报告](HANDOVER_20261106.md#模型权重)；Dyna-PPO 使用旧奖励，不纳入本轮同奖励对照。

## 测试设置

PPO、SAC、DQN 均在同一奖励环境中重新评估：标准参数/随机参数两种条件，测试种子 25042/26042，各取前 64 个环境的首回合；每模型每条件 128 回合。九个模型的初态与实际物理参数逐项配对。

先计算每回合指标，再求每模型每条件 128 回合的均值，最后报告三个训练种子的均值 ± 样本标准差。每方法每条件共 384 回合。训练曲线仅作学习诊断，正式结论以共同测试结果为准。

## 测评指标

主指标是**前向速度误差**，同时检查航向、侧移、跌倒和平滑性。RMSE 表示均方根误差，RMS 表示均方根。下表的 `mean` 对该回合的采样点取平均。

| 指标 | 计算方法 | 单位与判断 |
| --- | --- | --- |
| 前向速度误差（RMSE） | `√mean((vx−0.10)²)` | m/s，越低越好 |
| 平面速度误差（RMSE） | `√mean((vx−0.10)²+vy²)` | m/s，越低越好 |
| 航向误差（RMS） | 整个回合相对世界 +X 的角度误差均方根；角差折回 `[-π,π]` 后计算并换算为度 | °，越低越好 |
| 终点偏航 | 回合终点航向角误差的绝对值 | °，越低越好，与整个回合的航向误差不同 |
| 终点侧移 | `abs(y末−y初)` | m 或 cm，越低越好 |
| 前向位移 | `x末−x初` | m，10 秒目标约 1 m，并非越远越好 |
| 跌倒率 | 跌倒回合数 ÷ 总回合数 | %，越低越好 |
| 关节目标加速度（RMS） | 实际执行的关节目标位置二阶差分：`(q*_t−2q*_(t−1)+q*_(t−2))/Δt²`，再取 RMS | rad/s²，结合速度和位移判断平滑性 |
| 实际关节加速度（RMS） | 仿真关节速度一阶差分：`(q̇_t−q̇_(t−1))/Δt`，再取 RMS | rad/s²，结合速度和位移判断平滑性 |

`vx、vy` 为世界坐标系速度。`q*` 是经过动作裁剪、关节约束及时延/滞后处理后的关节目标位置；加速度按 **50 Hz（Δt=0.02 s）**采样，对时间和 8 个关节取均方根，排除重置跳变。这两项用于观察抖动程度，不是抖动频率或物理瞬时峰值；不能用不走路换取较低加速度。

跌倒判定为机身高度低于 **0.023 m**，或直立度低于 **0.5**（约整体倾斜超过 60°）。越界和无效状态另行记录，不自动计入跌倒。辅助记录关节目标变化率、动作裁剪比例、总回报和回合时长。

六次新训练及九模型共同评估均已完成，共 2304 个测试回合。结果目录：`runs/offpolicy_heading_20261107/`；训练日志在 `training/<方法>_seed<种子>/training.jsonl`。

## 结果

以下为三个训练种子的均值 ± 样本标准差；误差越低越好，10 秒目标位移约 1 m。

| 方法 | 标准参数速度误差 RMSE（m/s） | 随机参数速度误差 RMSE（m/s） | 随机参数航向误差 RMS（°） | 随机参数 10 秒位移（m） |
| --- | --- | --- | --- | --- |
| PPO | 0.0657 ± 0.0026 | 0.0361 ± 0.0013 | 4.80 ± 0.09 | 1.187 ± 0.036 |
| SAC | 0.0466 ± 0.0054 | 0.0440 ± 0.0014 | 6.55 ± 0.97 | 1.089 ± 0.026 |
| DQN | 0.0753 ± 0.0017 | 0.0610 ± 0.0046 | 14.87 ± 4.06 | 0.969 ± 0.095 |

**本轮结论：**SAC 的标准参数速度误差比 PPO 低 29.0%，侧移和前进超调也较小；但随机参数下的速度误差高 21.8%，航向控制也弱于 PPO。DQN 已能行走，随机参数速度误差高 69.0%，航向误差较大。三种方法在每种条件的 384 回合中均为 **0 次跌倒**，不能据此区分跌倒风险。

平滑性没有全面改善：SAC 标准参数下的关节目标加速度 RMS 为 532.16，PPO 为 479.74 rad/s²；随机参数下的实际关节加速度 RMS 为 302.09，PPO 为 284.21 rad/s²。DQN 随机参数下的关节目标加速度较低，但伴随较大的速度和航向误差。本轮保留 PPO 作为速度/航向基线，SAC 可继续研究鲁棒性与平滑性的取舍；这些结论限于本次固定参数和训练预算。

[完整指标报告](../runs/offpolicy_heading_20261107/analysis/summary.md) · [指标对照图](../runs/offpolicy_heading_20261107/analysis/performance.png) · [训练曲线](../runs/offpolicy_heading_20261107/analysis/training_curves.png) · [逐回合数据](../runs/offpolicy_heading_20261107/analysis/per_episode.csv)

### Demo

[PPO 视频](../runs/offpolicy_heading_20261107/videos/ppo_seed42/offpolicy_ppo_seed42_seed25042.mp4) · [SAC 视频](../runs/offpolicy_heading_20261107/videos/sac_seed42/offpolicy_sac_seed42_seed25042.mp4) · [DQN 视频](../runs/offpolicy_heading_20261107/videos/dqn_seed42/offpolicy_dqn_seed42_seed25042.mp4)

均为真实 Isaac RTX 录像，10 秒、720p、25 fps；固定训练种子 42、测试种子 25042，三段视频的实际初态和物理参数相同。单环境示例与批量统计分别采样，不能用这一回合替代上表。

## 代码与运行

- `sesame_rl/offpolicy.py`：SAC、DQN、GPU 回放及模型保存/加载。
- `sesame_rl/offpolicy_environment.py`：保存自动重置前的下一观测，奖励原样继承；超时继续 bootstrap，真实终止不 bootstrap。
- `scripts/probe_offpolicy.py`：GPU 数值验证；`scripts/train_offpolicy.py`：训练。
- `scripts/run_offpolicy_queue.py`：六次顺序训练、共同评估与报告生成。
- `scripts/evaluate_offpolicy.py`、`scripts/analyze_offpolicy.py`：配对评估、统计和曲线。
- `scripts/record_offpolicy_video.py`：固定样本的单模型录像；`scripts/run_offpolicy_videos.py`：评估完成后顺序录制三段 Demo。

训练和推理必须使用 CUDA 与激活后的 `env_isaaclab`。沿用交接文档的 `isaac_py` 函数，在新的结果目录运行：

```bash
SESAME_SAC_RUN="$(mktemp -d /home/ywwang/sesame/runs/sac_retrain_XXXXXX)"
isaac_py -u scripts/train_offpolicy.py --algorithm sac --seed 42 --run-dir "$SESAME_SAC_RUN"
```

DQN 将 `--algorithm` 改为 `dqn`，每次创建新目录。正式入口固定上述预算；开发短测需 `--smoke`。检查点保存模型、优化器、温度和随机数状态，但未保存完整回放与仿真状态，不能用于精确接续原环境轨迹。训练源码、上述最终模型和评估结果保留；旧权重、中间检查点和测试模型已按[清理记录](../reports/model_cleanup_20261108.json)删除。
