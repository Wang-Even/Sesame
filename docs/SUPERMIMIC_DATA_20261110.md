# Sesame 的 SuperMimic 数据准备

数据目录：[`datasets/supermimic_sesame_20261110/`](../datasets/supermimic_sesame_20261110/)。
本次准备的是 **Sesame 的参考动作与仿真交互数据**，用于后续适配 SuperMimic；尚未训练 SuperMimic 策略，也不是官方 G1 数据集。

## 已准备的数据

最终数量见 [`summary.json`](../datasets/supermimic_sesame_20261110/summary.json)，校验结果见 [`validation.json`](../datasets/supermimic_sesame_20261110/validation.json)。

| 划分 | 回合数 | 状态转移 | 参考轨迹 | 跌倒回合 |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 640 | 300,847 | 24 | 39 |
| 验证 | 640 | 299,406 | 8 | 42 |
| 测试 | 640 | 299,889 | 8 | 41 |
| 合计 | 1,920 | 900,142 | 40 | 122 |

交互数据累计约 5 小时仿真时长；筛选后的参考动作共 6 分钟。它们是多环境、多初态采样，并不等于 5 小时互不重复的新技能。

数据目录约 1.03 GiB。校验覆盖 4,068 个文件、全部回合和参考动作；动作裁剪、控制器记忆、延迟/滞后和前后观测均通过复算，控制器数值最大绝对误差约 `2.4e-7`。参考动作来自真实记录，逐帧匹配原始片段。

- **参考动作**：从已有 PPO、SAC 策略的仿真回合中筛选，去掉前 1 秒启动阶段，每条保留连续 9 秒、451 帧。关节顺序为 `[R1,R2,L1,L2,R4,R3,L3,L4]`，采样频率 50 Hz。
- **动力学数据**：包含 0.10 / 0.15 / 0.20 m/s 指令、启停、动作扰动、倾斜初态恢复。保留成功与失败回合，记录终止前状态。
- **随机化信息**：每回合的质量、摩擦、控制增益、力矩/速度限制、延迟和滞后参数，以及实际仿真属性读回值。
- **历史轨迹**：18 条旧评估轨迹单独放在 `legacy/`；它们缺少完整控制器状态，不纳入新训练集。

参考动作的筛选条件在采集前固定：完整 10 秒且无跌倒、越界或无效状态；去掉前 1 秒后，平面速度 RMSE ≤ 0.05 m/s、平均前进速度误差 ≤ 0.025 m/s、航向 RMS ≤ 10°、最大航向偏差 ≤ 20°、横向位移 ≤ 0.12 m、目标关节加速度 RMS ≤ 250 rad/s²、实际关节加速度 RMS ≤ 400 rad/s²；全回合关节限位越界量 ≤ 0.02 rad。这些是本数据集的工程筛选条件，不是课程官方成功标准。

**0.15 和 0.20 m/s 的尝试未通过筛选，保留为动力学数据，不作为高速参考动作。**现有策略只在 0.10 m/s 上训练，这一结果不能证明机器人本体无法高速行走。下一步需训练或优化高速控制器，再按相同流程收集。

## 文件与字段

| 文件/目录 | 用途 |
| --- | --- |
| `splits/{train,validation,test}.json` | 各划分的动力学回合与参考动作清单 |
| `episodes/`、`supplemental/` | 完整回合 NPZ、参数 JSON、采集源码快照与日志 |
| `references/` | 通过筛选的参考动作 NPZ |
| `schema.json`、`protocol.json` | 字段定义、坐标系、初始采集设置与筛选阈值；补采设置见各补采目录的 `protocol.json` |
| `combined_episode_index.json`、`reference_index.json` | 每条数据的来源、指标、拒收原因和 SHA256 |
| `checksums.json`、`validation.json` | 文件校验与 CUDA 数据验证结果 |

完整回合的 `state_*` 数组有 **T+1 帧**；动作、指令和终止标记有 **T 步**。`state[t+1]` 是执行 `action[t]` 后、自动重置前的状态。保存机身、关节、刚体及足端运动，以及执行目标、限速后的目标、三帧延迟队列和上一执行动作。

位置使用世界坐标轴并减去各环境原点；速度仍为世界坐标轴表示；四元数顺序为 `wxyz`，角度单位为 rad。足端是 URDF 中四个固定足端坐标系的原点。

接触力是各刚体在控制周期末的法向接触合力，不等于精确足尖接触，也不是整个控制周期的积分。关节力矩是隐式执行器的估计值。`legacy_reward` 保留旧环境固定 0.10 m/s 的奖励定义，**不能当作变速任务评分或模仿奖励**；后续训练应根据参考动作重算奖励。

## 使用方式

从项目根目录运行，所有模型、仿真及数值验证均使用 CUDA：

```bash
cd /home/ywwang/sesame
isaac_py() {
  CONDA_NO_PLUGINS=true PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/ubuntu/miniforge3/bin/conda run -n env_isaaclab --no-capture-output \
  /home/ubuntu/miniforge3/envs/env_isaaclab/bin/python "$@"
}
isaac_py scripts/validate_supermimic_dataset.py datasets/supermimic_sesame_20261110
```

在使用 CUDA 的训练程序中：

```python
from sesame_rl.motion_dataset import SesameMotionDataset, SesameReferenceDataset

path = "datasets/supermimic_sesame_20261110"
transitions = SesameMotionDataset(path, split="train", horizon=8)
batch = transitions.sample(64)
# state: [64, 9, 77]；executed_action: [64, 8, 8]；context: [64, 9]

references = SesameReferenceDataset(path, split="train", future_frames=10)
motion = references.sample(64)
# current["joint_pos"]: [64, 8]；future["joint_pos"]: [64, 10, 8]
```

77 维状态是世界模型的示例输入，包含机身/关节状态和控制器记忆，**不是策略的观测维度**；9 维 `context` 是随机化参数。原策略观测仍为 36 维，未来模仿策略还需接入参考动作。足端、刚体和接触数据保留在 NPZ 中，可用于辅助监督。

窗口不会跨回合或重置；参考序列不会循环拼接。短于窗口长度的回合会被读取器跳过，失败回合仍保存在原始数据中，可用更短窗口学习终止。默认按可用窗口数采样，训练时可按需要增加失败附近窗口的采样比例。

122 个跌倒回合中，67 个至少包含 8 步，另外 55 个不足 8 步。默认 8 步读取器不会选中这 55 个短回合；学习短时失稳或终止时可另用 `horizon=1`，不能把它们补零后当成真实连续轨迹。

训练、验证、测试按完整回合和采样种子划分，同一初态的 PPO/SAC 配对留在同一划分。归一化参数只能用 train 拟合。三个划分共享策略来源和步态类型，因此测试用于检查不同初态与随机化条件下的表现，不能代表新技能泛化。

## 重新采集

使用新目录；保留当前数据和旧实验。

```bash
SESAME_DATA_DIR="$(mktemp -d /home/ywwang/sesame/datasets/supermimic_new_XXXXXX)"
isaac_py scripts/collect_supermimic_data.py --output-dir "$SESAME_DATA_DIR" --envs-per-profile 16
isaac_py scripts/collect_supermimic_data.py --output-dir "$SESAME_DATA_DIR/supplemental/recovery" --envs-per-profile 32 --recovery-only
isaac_py scripts/collect_supermimic_data.py --output-dir "$SESAME_DATA_DIR/supplemental/extra_reference" --envs-per-profile 32 --reference-only
isaac_py scripts/prepare_supermimic_dataset.py "$SESAME_DATA_DIR" \
  --supplement "$SESAME_DATA_DIR/supplemental/recovery" \
  --supplement "$SESAME_DATA_DIR/supplemental/extra_reference"
isaac_py scripts/validate_supermimic_dataset.py "$SESAME_DATA_DIR"
```

打包只做文件转换与元数据整理；采集、策略推理、运动学计算和数值校验在 GPU 上执行。新的倾斜恢复初态在首帧记录前设置，不在回合中施加未记录的推力。

当前参考动作仍继承来源策略的步态特点，未做真机验证；尚缺合格的高速、转向参考动作。后续需要实现参考目标接口、多步世界模型及可微策略更新，才能开始 SuperMimic 训练。
