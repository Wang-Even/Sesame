# Sesame 平地行走 PPO 基线

**权重状态（2026-11-08）：**本页旧 PPO 的最终权重和中间检查点已删除，历史指标、日志和视频保留；重新评估或录像需恢复权重或在新目录重新训练。当前保留的模型见[模型权重](HANDOVER_20261106.md#模型权重)。

本工程使用现有 Isaac Sim 4.5 / Isaac Lab 2.3 和 RSL-RL 3.1.2。任务基于已导入的 Sesame ML 参考模型创建；课程统一环境尚未提供，因此这里的回报是本地基线指标。

2026-11-02 已完成默认配置训练（2,048,000 条转移，约 5 分 45 秒含评测）。最终检查点原路径为 `runs/ppo_baseline_20261102_seed42/final.pt`（已删除）。32 个种子 1042 的配对初始状态评测中，PPO 平均 10 秒前进 1.2772 m（标准差 0.0231 m），跌倒 0/32；零残差控制位移约为 0。完整指标、逐回合数据和学习曲线均保存在该运行目录。

已确认的基线局限：首个评测回合约 2 秒后偏航至 −88.5°，随后机身基本横向沿目标世界方向移动。当前奖励没有累计航向误差项，因此高位移不代表低偏航直行；同一回合原始动作的 32.2% 分量被边界裁剪。保留本次配置与结果，后续可将航向约束作为明确的改进/消融项。

## 任务与控制

环境实现：`sesame_rl/environment.py`。机器人具有浮动基座、8 个转动关节，目标沿世界坐标 `+x` 方向以 0.1 m/s 前进。每回合最多 10 秒。

物理步长为 0.002 秒，每 10 个物理步执行一次策略动作，即控制频率 50 Hz。动作是固件顺序 `R1 R2 L1 L2 R4 R3 L3 L4` 的 8 维站立姿态偏移，限幅到 `[-1,1]` 后乘以 0.58 rad，再经过每关节位置限位和 10.472 rad/s 的目标角变化限制。关节力矩上限为 0.18 N·m。参考驱动刚度 5、阻尼 0.0025，均为公开模型参数，尚未实机标定。

36 维观测依次为机身坐标系线速度（乘 5）、角速度（乘 0.25）、重力方向、机身坐标系前进指令（乘 5）、相对站姿的 8 个关节角、除以最大关节速度的 8 个关节速度，以及上一动作。输入还使用随训练保存的观测归一化统计。

站姿重置使用源模型的关节角，基座高度 0.049 m，并对初始关节角添加 ±0.015 rad 的随机扰动。若基座低于 0.023 m、倾斜超过 60°、离开局部 ±3 m 区域或出现非有限状态则终止。真正终止优先于时间截断，防止在最后一步跌倒后错误进行价值自举。

地面摩擦系数为 0.9；机器人与地面接触偏移为 1 mm，静止偏移为 0。碰撞体在生成场景时取消实例化以确保参数实际生效，视觉网格仍保留实例化。源 USD 没有被修改。

## 奖励

速度跟踪和向前速度奖励均乘以直立程度的平方。每步将连续奖励率乘以 0.02 秒，跌倒/越界/无效状态另外扣除一次终止惩罚。

| 奖励项 | 系数/定义 |
| --- | --- |
| 前进速度跟踪 | `1.5 * exp(-((vx - 0.1)^2 + vy^2) / 0.01)` |
| 向前运动 | `clip(vx / 0.1, -1, 1)` |
| 直立 | `0.35 * upright` |
| 横向速度 | `-0.10 * (vy / 0.1)^2` |
| 偏航角速度 | `-0.03 * yaw_rate^2` |
| 滚转/俯仰角速度 | `-0.01 * sum(roll_pitch_rate^2)` |
| 垂直速度 | `-0.02 * (vz / 0.1)^2` |
| 力矩 | `-0.02 * mean((torque / 0.18)^2)` |
| 动作变化 | `-0.08 * mean((action - previous_action)^2)` |
| 站姿偏离 | `-0.10 * mean((q - q_stand)^2)` |
| 真正终止 | 一次扣除 2 |

## PPO 设置

| 参数 | 默认值 |
| --- | --- |
| 并行环境 | 256 |
| 每环境 rollout 步数 | 32 |
| 更新次数 | 250 |
| 环境转移数 | 2,048,000 |
| 随机种子 | 42 |
| Actor / Critic | 独立 MLP，128、128、64，ELU |
| 优化器 | Adam，初始学习率 3e-4，按 KL 自适应 |
| PPO clip | 0.2 |
| Gamma / GAE lambda | 0.99 / 0.95 |
| 每次更新 | 5 epochs，4 mini-batches |
| 初始动作标准差 | 0.5，log-std 参数化 |
| 熵系数 | 0.005 |
| 梯度范数上限 | 1.0 |
| 保存间隔 | 每 50 次更新，另存 final.pt |

训练、仿真、策略前向和反向均在 `cuda:0` 上执行。源码快照和 SHA-256 替代 Git 日志；代码在导入 RSL-RL 前阻止 GitPython 初始化，并禁用其 Git 快照函数。

## 启动

```bash
cd /home/ywwang/sesame
CONDA_NO_PLUGINS=true PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/miniforge3/bin/conda run -n env_isaaclab --no-capture-output \
  /home/ubuntu/miniforge3/envs/env_isaaclab/bin/python -u scripts/train_ppo.py \
  --num-envs 256 --iterations 250 --steps-per-env 32 \
  --seed 42 --eval-seed 1042 --eval-episodes 32 \
  --run-dir runs/ppo_baseline_20261102_seed42
```

每次运行需要新的结果目录，避免覆盖已完成的训练。训练与评测脚本带有 Isaac 退出超时保护；结果写盘后若清理挂起，只结束当前进程。

## 输出与评测

- `final.pt`：策略、价值网络、观测归一化、优化器和训练步数。
- `model_*.pt`：周期性检查点。
- `environment_config.json` / `ppo_config.json`：完整任务和算法配置。
- `sources/`：此次运行源码及资产快照。
- `training.jsonl` / TensorBoard events：每次更新的回报、回合长度、位移、跌倒率和损失。
- `evaluation_ppo.json` / `evaluation_zero_residual.json`：重新加载后的 PPO 策略与零残差站姿控制的配对评测。
- `eval_trajectory_*.json`：首个评测回合逐步状态与动作。
- `result.json`：完成状态、训练规模、评测汇总和检查点校验信息。
- `videos/ppo_walk_seed1042.mp4`：最终策略的完整 10 秒实时仿真录像，1280×720、25 fps、H.264；对应 JSON 保留检查点哈希与该录像回合结果。

评测使用种子 1042，对两种控制器采用完全相同的初始状态，分别记录选定环境的第一个完整回合，共 32 回合；避免选取较早结束或较好的轨迹。统计在自动重置之前采集，含前向位移、平均前向速度、回报、跌倒率和持续时间。保存后在 GPU 上重新构建策略并加载权重，检查动作与保存前完全一致。

不把存活或较高训练回报单独解释为学会行走；应同时查看前进位移、速度、跌倒和轨迹。此基线还没有加入课程评测规则、真机延迟随机化或实际硬件反馈接口。

元数据绘图（不加载策略）：

```bash
python scripts/plot_training.py runs/ppo_baseline_20261102_seed42
```

以下为原录像命令；恢复旧权重后方可执行（CUDA 推理与仿真，NVENC 视频编码）：

```bash
CONDA_NO_PLUGINS=true PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/miniforge3/bin/conda run -n env_isaaclab --no-capture-output \
  /home/ubuntu/miniforge3/envs/env_isaaclab/bin/python scripts/record_ppo_video.py \
  --run-dir runs/ppo_baseline_20261102_seed42 --seed 1042
```
