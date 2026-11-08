# Sesame PPO：多种子域随机化与时延消融实验

**权重状态（2026-11-08）：**本页旧 PPO、完整域随机化和固定延迟消融的 9 个最终模型及中间检查点已删除，历史指标、日志和视频保留。当前保留的模型见[模型权重](HANDOVER_20261106.md#模型权重)。

实验方案于 2026-11-03（Asia/Shanghai）在新测试前固定。机器上的物理仿真、策略训练与评测均在 CUDA 执行；图表和报告仅处理保存的元数据。

## 已完成结果

三种方法 × 三个训练种子全部完成，每份模型训练 2,048,000 条转移，其中两份历史种子 42 模型复用、新训练七份。正式评测共 18 个案例、2,304 回合，所有回合均完成 10 秒；未出现跌倒、越界或非有限状态。以下误差条均为三个训练种子均值的样本标准差，独立训练重复数为 **n=3**。

| 条件 | 原 PPO：速度 RMSE（m/s） | 完整 DR：速度 RMSE（m/s） | 固定时延 DR：速度 RMSE（m/s） |
| --- | --- | --- | --- |
| 标准参数 | 0.04836 ± 0.01141 | 0.06217 ± 0.00300 | 0.10037 ± 0.00614 |
| 随机参数 | 0.06943 ± 0.00556 | 0.03748 ± 0.00172 | 0.04172 ± 0.00238 |

这里的 RMSE 来自每个回合的 500 个控制步，再对回合和训练种子分别汇总，不是用回合平均速度计算的误差。三种方法在随机条件下的平均回合位移依次为 0.6727、1.1343、1.1645 m；平均终点绝对偏航依次为 51.17°、13.10°、11.89°。

### 对研究问题的回答

1. **整套 DR 方案改善了训练范围内的速度跟踪鲁棒性。** 随机条件下主指标比原 PPO 降低 46.03%，三个训练种子均改善。平均终点绝对横向位移也由 0.0959 m 降到 0.0354 m。
2. **时延/响应随机化提供了有限但可观察的额外收益。** 完整 DR 比固定时延消融组的随机条件 RMSE 平均降低 10.17%。种子 42、43 改善，种子 44 几乎持平，不能据此声称普遍或统计显著的优势。
3. **存在标准参数条件的性能代价。** 完整 DR 的标准条件 RMSE 比原 PPO 增加 28.57%，回报由 22.424 降至 17.828，平均绝对侧移由 0.0311 m 增至 0.1484 m。标准条件恢复原控制器的零附加滞后/延迟，与 DR 训练中的 20–60 ms 响应滞后不同。
4. **这轮没有证明降低跌倒率或整体减少抖动。** 三组所有回合均零跌倒。随机条件下完整 DR 的下发目标角变化率 RMS 由 3.131 增至 4.189 rad/s，目标角二阶差分 RMS 由 109.62 增至 133.24 rad/s²；实际关节加速度 RMS 均值从 300.87 降至 284.71 rad/s²，但种子 43 反而增加。原始动作裁剪比例从 35.25% 增至 43.92%。各指标对应不同物理量，不能统称“抖动降低”。

时延消融的随机条件主指标逐种子结果如下：

| 训练种子 | 原 PPO | 完整 DR | 固定时延 DR | 完整 DR − 固定时延 DR |
| --- | --- | --- | --- | --- |
| 42 | 0.07523 | 0.03567 | 0.04227 | −0.00660 |
| 43 | 0.06894 | 0.03766 | 0.04378 | −0.00612 |
| 44 | 0.06413 | 0.03910 | 0.03911 | −0.00001 |

单位均为 m/s，越低越好。目标速度为 0.10 m/s，走得更远不等于跟踪更准确；固定时延组在标准条件下平均走 1.744 m，但速度 RMSE 最大。

### 数据与图表

- [完整统计与结果表](../runs/robustness_ablation_20261103/analysis/summary.md)
- [逐训练种子 CSV](../runs/robustness_ablation_20261103/analysis/per_training_seed.csv)
- [组均值/标准差 CSV](../runs/robustness_ablation_20261103/analysis/group_summary.csv)
- [配对种子差值 CSV](../runs/robustness_ablation_20261103/analysis/paired_seed_differences.csv)
- [性能对比图](../runs/robustness_ablation_20261103/analysis/performance.png)
- [平滑性与动作裁剪图](../runs/robustness_ablation_20261103/analysis/smoothness.png)
- [多种子学习曲线](../runs/robustness_ablation_20261103/analysis/learning_curves.png)
- [全部逐回合数据](../runs/robustness_ablation_20261103/evaluation/cases/)
- [训练预算与重载检查](../runs/robustness_ablation_20261103/training_validation.json)
- [独立统计与溯源审计](../runs/robustness_ablation_20261103/analysis/independent_review.json)
- [九份旧检查点的历史路径与哈希（权重已删除）](../runs/robustness_ablation_20261103/resolved_training_checkpoints.json)
- [实验完成记录](../runs/robustness_ablation_20261103/result.json)

九份检查点均通过保存后重载，动作最大误差为 0；所有模型与源码/资产哈希验证通过。四套测试初态（两条件 × 两评测种子）在九个策略之间逐项一致，包括根状态、关节位置/速度和 PhysX 实际参数；36 次域参数回读均通过。GPU 探针独立重算指标，最终统计还进行了独立复核。

训练曲线的实线为每个种子按 10 次更新平滑后取三种子的均值，阴影为样本标准差。三组训练域不同，固定时延组较高的训练回报不能代替相同测试条件下的结果。

### 视频

- [三组并排对照视频：10 秒，1920×400，25 fps](../runs/robustness_ablation_20261103/videos/ppo_study_comparison_seed3042.mp4)
- [原 PPO 原始录像：720p](../runs/robustness_ablation_20261103/videos/baseline_seed42/ppo_study_baseline_seed42_seed3042.mp4)
- [完整 DR 原始录像：720p](../runs/robustness_ablation_20261103/videos/full_dr_seed42/ppo_study_full_dr_seed42_seed3042.mp4)
- [固定时延 DR 原始录像：720p](../runs/robustness_ablation_20261103/videos/fixed_delay_seed42/ppo_study_fixed_delay_seed42_seed3042.mp4)
- [视频参数、初态配对与编码验证](../runs/robustness_ablation_20261103/videos/comparison_validation.json)

三段原始录像都使用预先指定的训练种子 42 和测试种子 3042，在相同完整随机化测试条件下录制；实际初态与物理域快照严格一致。三个回合均完成 10 秒、无跌倒，前向位移依次为 0.4602、1.0317、1.0140 m。视频是单环境演示，不能代替跨三训练种子的 2,304 回合统计；其 CUDA 抽样形状与批量测试不同，因此同一种子不代表复现批量测试的第一个回合。

### 复现入口

以下命令为历史运行记录。旧权重已删除，原训练队列和对比评估不能直接重跑；需恢复原权重，或在新目录从头训练并建立新协议。已有统计与视频仍可查看。

已有完整训练结果会由队列校验并跳过；失败或进行中的目录会报错，避免覆盖。正式评测的 `--resume` 仅在协议、源码、配置和模型哈希全部一致时复用已完成案例。

```bash
cd /home/ywwang/sesame
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_ppo_study_queue.py \
  --study-dir runs/robustness_ablation_20261103

CONDA_NO_PLUGINS=true PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/miniforge3/bin/conda run -n env_isaaclab --no-capture-output \
  /home/ubuntu/miniforge3/envs/env_isaaclab/bin/python scripts/evaluate_ppo_study.py \
  --protocol runs/robustness_ablation_20261103/protocol.json \
  --output-dir runs/robustness_ablation_20261103/evaluation --resume

PYTHONDONTWRITEBYTECODE=1 python3 scripts/analyze_ppo_study.py \
  runs/robustness_ablation_20261103
```

从头独立重跑应建立新的实验目录和协议副本，保持所有方法、种子、预算与最终测试设置不变，避免覆盖本轮记录。录像复现入口为 `scripts/record_study_comparison.py`，应指定新的 `--output-dir` 保存另一份录像。

## 研究问题

在相同 PPO、网络、奖励和训练预算下，完整域随机化能否改善留出动力学条件下的逐时刻前向速度跟踪？将指令延迟和响应时间常数固定为相同范围的中心后，这种改善是否仍然存在？

| 组别 | 训练方法 |
| --- | --- |
| `baseline` | 原固定参数 PPO；使用原初始化和控制器 |
| `full_dr` | 摩擦、质量/惯量、舵机增益/限幅、响应滞后、延迟和初态随机化 |
| `fixed_delay` | 与完整 DR 一致，仅将响应时间常数固定为 0.04 s，延迟固定为 1 个控制步（0.02 s） |

`fixed_delay` 仍消费与完整 DR 相同的随机抽样，再覆盖这两个控制参数，避免连带改变其他物理参数的随机样本。它保留完整 DR 的初始状态分布。

`baseline` 与 `full_dr` 的对比衡量整个训练方案，包括初态扰动和平均执行器动力学的变化；不能解释为纯粹“随机性”的因果作用。`full_dr` 与 `fixed_delay` 的对比针对滞后/延迟变化，不能进一步区分二者各自的贡献。

## 固定预算与训练重复

每组训练种子为 42、43、44。每次从头训练 256 个并行环境、32 步 rollout、250 次迭代，共 2,048,000 条环境转移。PPO 配置与历史实验一致，使用最终检查点，不依据最终测试挑选检查点。

已有的原 PPO 种子 42 和完整 DR 种子 42 模型满足相同预算，直接复用；另训练 7 个模型。原模型、资产和环境源码保持不变。

固定协议与源码哈希：[protocol.json](../runs/robustness_ablation_20261103/protocol.json)、[protocol.sha256](../runs/robustness_ablation_20261103/protocol.sha256)。

## 最终测试

- 所有策略使用同一个评测环境，分别在标准参数和完整随机参数条件下测试。
- 每个策略、每种条件使用新的种子 3042、4042，各取 256 个并行环境中前 64 个环境的首个完整回合，共 128 回合。
- 各策略使用相同初态和实际物理域样本，并逐项验证一致性。种子 2042 的既有结果只作为开发信息。
- 使用确定性均值动作和各检查点保存的观测归一化，不继续更新策略。
- 回合最长 10 秒，控制频率 50 Hz；地形、奖励、动作约束和终止条件不变。

主指标是每回合逐控制步的世界坐标前向速度误差 RMSE，目标为 0.10 m/s。先在每个检查点的 128 回合上取均值，再统计三个训练种子均值的平均值与样本标准差。训练重复数为 **3**，不能把 384 个评测回合视为 384 次独立训练。

辅助指标包括二维速度误差、前向位移、回合平均速度误差、终点绝对偏航、终点绝对横向位移、跌倒率和时长，以及实际下发目标角变化/二阶差分和实际关节速度差分。平滑性统计在相同控制采样频率下计算，排除自动重置跳变。动作裁剪比例单独报告，不作为抖动的替代指标。

新增指标已通过 GPU 独立重算验证，包括 500 个控制样本的完整回合、连续三次自动重置、仅重置部分环境，以及一控制步回合中目标角二阶差分尚未定义时保留空值。实际关节加速度是 50 Hz 采样的关节速度有限差分，不表示 500 Hz 物理步内的峰值。验证报告见 [study_environment_probe.json](../reports/study_environment_probe.json)。

记录每个训练种子的结果、跨种子标准差以及相同训练种子的组间差值。只报告观察到的效果与权衡，不因三个训练种子的均值排序宣称普遍优越。

## 结果范围

当前实验是本地参考仿真中的分布内动力学泛化评测，不是课程官方评分环境，不代表真机校准或真机性能。课程统一环境取得后仍需按其固定规则重新评测。

代表视频预先选定训练种子 42、测试种子 3042、随机参数条件下的首个完整回合。单环境渲染与批量抽样可能产生不同具体物理样本，因此会记录实际域参数；不根据视频表现挑选种子或回合。
