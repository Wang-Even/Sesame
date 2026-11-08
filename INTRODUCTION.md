# Sesame 机器人资产与 Isaac 环境检查

报告入口（更新于 2026-11-08）：[简版实验报告](docs/HANDOVER_20261106.md) · [浏览版](docs/HANDOVER_20261106.html)。聚焦代码结构与使用、奖励函数、已完成 Demo、基线结果与测评指标，以及待改进问题和对应方法。下文保留此前阶段的记录。

模型入口：[当前保留的 12 个最终权重](docs/HANDOVER_20261106.md#模型权重)。2026-11-08 已清理旧 PPO、旧域随机化、固定延迟消融的权重及中间/测试检查点；历史指标、日志和视频保留，旧对比不能直接重新评估。PPO、SAC、DQN、Dyna-PPO 各保留 3 个最终模型，详见[清理记录](reports/model_cleanup_20261108.json)。

新增实验（2026-11-07）：[同奖励 PPO / SAC / DQN 对照](docs/OFFPOLICY_STUDY_20261107.md)，已完成三种子训练与 2304 回合共同评估，包含结果指标、训练曲线和运行方法。

更新日期：2026-11-03（Asia/Shanghai）。

已阅读课程 PDF、导入机器人资产，并在 Isaac GPU 环境完成 PPO 平地行走基线、域随机化和多种子时延消融实验。每次训练均为 256 个并行环境、250 次更新、2,048,000 条环境转移。课程统一训练/评测环境尚未取得，这里的结果来自本地参考任务。

## 多种子对照与时延消融实验（最新）

已完成原 PPO、完整域随机化（DR）和固定时延 DR 三种方法，训练种子均为 42、43、44，共九份最终模型；两份历史种子 42 模型复用，新增七次完整训练。每次预算均为 2,048,000 条转移。测试种子预先固定为 3042、4042，共完成 2,304 个配对回合。

主指标为逐控制步前向速度跟踪 RMSE，单位 m/s，越低越好；表中为三个训练种子均值 ± 样本标准差。

| 测试条件 | 原 PPO | 完整 DR | 固定时延 DR |
| --- | --- | --- | --- |
| 标准参数 | 0.04836 ± 0.01141 | 0.06217 ± 0.00300 | 0.10037 ± 0.00614 |
| 随机参数 | 0.06943 ± 0.00556 | 0.03748 ± 0.00172 | 0.04172 ± 0.00238 |

完整 DR 在随机条件下相对原 PPO 降低误差 46.03%，三个训练种子均改善；相对固定时延消融组降低 10.17%，其中两个种子改善、一个基本持平。代价是标准条件下相对原 PPO 的误差增加 28.57%。三组全部零跌倒，平滑性指标没有一致改善，不能把结果解释为降低跌倒率或整体减少抖动。

- [实验方法、消融、结果解释与局限](docs/ROBUSTNESS_STUDY.md)
- [三组策略并排行走视频](runs/robustness_ablation_20261103/videos/ppo_study_comparison_seed3042.mp4)
- [性能对比图](runs/robustness_ablation_20261103/analysis/performance.png)
- [平滑性对比图](runs/robustness_ablation_20261103/analysis/smoothness.png)
- [多种子学习曲线](runs/robustness_ablation_20261103/analysis/learning_curves.png)
- [结果表与 CSV](runs/robustness_ablation_20261103/analysis/summary.md)
- [实验协议](runs/robustness_ablation_20261103/protocol.json)

正式课程评分仍需课程统一环境；本次为本地参考任务、训练范围内的物理扰动测试，独立训练重复数为 n=3。

## 域随机化 PPO 初步结果（单训练种子）

探索运行：[runs/ppo_dr_20261102_seed42](runs/ppo_dr_20261102_seed42)。2026-11-02 21:17–21:23（Asia/Shanghai）完成训练与配对评测，总耗时约 5 分 50 秒。每回合随机化摩擦、质量与惯量、驱动增益、力矩/速度限幅、响应滞后、指令延迟和初态；地形仍为平地，奖励和 PPO 配置与原基线一致。具体范围见 [域随机化说明](docs/DOMAIN_RANDOMIZATION.md)。

以下均使用留出的种子 2042，每个条件和策略各 64 个完整的 10 秒回合。相同条件下的三种策略严格配对初始状态与实际物理参数；零残差控制位移约为 0，表内列出两种学习策略。± 表示回合间标准差。

| 评测条件 | 策略 | 前向位移（m） | 平均终点绝对偏航 | 跌倒回合 | 平均回报 |
| --- | --- | --- | --- | --- | --- |
| 标准参数 | 原 PPO | 1.2771 ± 0.0276 | 87.7° | 0 / 64 | 21.699 |
| 标准参数 | DR PPO | 1.2480 ± 0.0256 | 10.4° | 0 / 64 | 18.114 |
| 随机参数 | 原 PPO | 0.5932 ± 0.1922 | 82.3° | 0 / 64 | 14.951 |
| 随机参数 | DR PPO | 1.1069 ± 0.0832 | 15.5° | 0 / 64 | 23.036 |

本次运行中，随机参数下平均位移提高 86.6%，偏航也减小；标准参数下位移减少 2.3%，回报下降。新策略仍有偏航和侧移，随机参数评测中 42.8% 的原始动作分量被裁剪。以上仅覆盖一个训练种子和已定义随机化范围内的仿真条件，不代表真机验证或所有域都更优。

- 旧 DR 最终检查点 `runs/ppo_dr_20261102_seed42/final.pt`（已删除）
- [DR 随机参数行走视频：10 秒，720p，25 fps](runs/ppo_dr_20261102_seed42/videos/randomized/ppo_dr_randomized_seed2042.mp4)
- [DR 标准参数行走视频：10 秒，720p，25 fps](runs/ppo_dr_20261102_seed42/videos/nominal/ppo_dr_nominal_seed2042.mp4)
- [学习曲线](runs/ppo_dr_20261102_seed42/plots_dr/dr_learning_curves.png)
- [配对评测对比图](runs/ppo_dr_20261102_seed42/plots_dr/dr_evaluation_comparison.png)
- [完整结果与六组评测汇总](runs/ppo_dr_20261102_seed42/result.json)
- [物理参数读回、随机种子与局部重置检查](reports/domain_randomization_probe.json)
- [随机化范围、训练/评测/录像命令](docs/DOMAIN_RANDOMIZATION.md)

本轮实验当时保持原环境及原基线检查点不变。新检查点重新加载后的策略动作与保存前完全一致，训练前后及每组评测的实际 PhysX 参数读回均通过验证。

两段视频均为最终策略在 Isaac 中实时运行的完整回合，使用种子 2042，没有挑选替换回合。标准参数录像前进 1.2144 m、终点偏航 7.4°；随机参数录像前进 1.0849 m、终点偏航 8.0°；均未跌倒。录像为单环境，同一种子下的 CUDA 批量采样形状与 256 环境评测不同，不能视为复现表中第一个回合；各自 JSON 报告保留实际域参数与初态。相机固定世界朝向，仅跟随平移，地面网格和箭头不参与碰撞。

## PPO 基线结果

运行目录：[runs/ppo_baseline_20261102_seed42](runs/ppo_baseline_20261102_seed42)。训练与评测于 2026-11-02 16:58–17:04（Asia/Shanghai）完成，总耗时约 5 分 45 秒。训练种子 42，评测种子 1042。

最终策略在 32 个配对初始状态下评测，每回合 10 秒；重新加载检查点后的策略输出与保存前完全一致。

| 指标 | 零残差站立控制 | PPO 最终策略 |
| --- | --- | --- |
| 平均前向位移 | 约 0 m | 1.2772 m |
| 位移标准差 | 约 0 m | 0.0231 m |
| 平均前向速度 | 约 0 m/s | 0.1277 m/s |
| 跌倒回合 | 0 / 32 | 0 / 32 |
| 完成 10 秒回合 | 32 / 32 | 32 / 32 |
| 本地任务平均回报 | 9.0184 | 21.7280 |

- 旧 PPO 最终检查点 `runs/ppo_baseline_20261102_seed42/final.pt`（已删除）
- [PPO 仿真视频：10 秒，720p，25 fps](runs/ppo_baseline_20261102_seed42/videos/ppo_walk_seed1042.mp4)
- [视频录制参数与结果](runs/ppo_baseline_20261102_seed42/videos/ppo_walk_seed1042.json)
- [学习曲线](runs/ppo_baseline_20261102_seed42/learning_curve.png)
- [首个配对回合位移曲线](runs/ppo_baseline_20261102_seed42/evaluation_trajectory.png)
- [完整运行结果](runs/ppo_baseline_20261102_seed42/result.json)
- [逐回合评测](runs/ppo_baseline_20261102_seed42/evaluation_ppo.json)
- [任务、奖励、超参数与运行命令](docs/PPO_BASELINE.md)

这版仍有明显航向问题：保存的首个评测回合在约 2 秒内偏航至 −88.5°，之后基本横向沿世界 `+x` 移动。奖励约束世界坐标速度并惩罚偏航角速度，没有约束累计航向误差，因此不能称为低偏航直行步态。该回合 32.2% 的原始动作分量被裁剪到合法边界；下发动作仍经过位置和限速约束。

这是单次训练种子的仿真基线，评测扰动仅覆盖初始关节角的小范围变化；这些结果不代表多训练种子统计或真机表现。

视频使用最终检查点和训练时的观测归一化，在单环境中运行种子 1042 的完整策略回合；相机保持世界朝向，标出位移、航向和 0.1 m 地面刻度。该录像回合前进 1.2932 m、未跌倒，保留了先转身后侧向移动的过程。单环境录像与上述 256 并行环境中的 32 回合评测有轻微数值差异，未修改任务、策略或机器人动力学。

## PDF 要求

原文件：`大作业选题：Sesame 四足机器人行走 (1).pdf`；提取全文见 [docs/course_requirements.txt](docs/course_requirements.txt)。

- 机器人为 ESP32 小型四足，8 个舵机，每条腿 2 个自由度。
- 仿真任务是从标定站立姿态开始，持续沿指定前方行走，减少跌倒、偏航与关节抖动；建议使用 PPO。
- 动作为 8 维连续关节目标角度偏移，须经过统一缩放、限位和限速。实际关节角度不等于目标角度。
- 观测可包含姿态/重力方向、机身速度、关节位置与速度、上一动作；具体字段、坐标系、单位和归一化应以课程接口为准。
- 100 万至 300 万条环境转移是初始训练规模参考，不是最低提交要求。并行环境转移累加，物理子步不单独计数。
- 期中完成 PPO 基线、改进、对照与至少一项消融，并按课程统一环境重新评测。训练日志中的自定义奖励不能代替正式评测回报。
- 期末做短时真机 Demo。策略依据同步预测模型的状态运行，真机不提供完整运动状态反馈；需保持训练和部署的关节顺序、单位、控制周期、动作处理及归一化一致。

PDF 说明课程组提供模型、起点代码、统一评测与部署接口，但 PDF 本身没有这些文件或可点击下载地址。机器上找到的 `/home/ubuntu/jrWork/rl-course-sesame-project` 也是课程文档与导出材料，未发现对应可执行训练工程。

## 已导入的官方资产

来源：[dorianborian/sesame-robot](https://github.com/dorianborian/sesame-robot)。使用官方 HTTP ZIP 下载，没有执行 Git 命令。

- 提取目录：[assets/upstream/sesame-robot](assets/upstream/sesame-robot)
- 原始归档：`downloads/sesame-robot-main.zip`（87,884,909 字节，约 83.8 MiB）
- ZIP 内标注的版本：`f328ed35f5cbcbdd44a5e16e3ec679d2caa00d92`
- 归档 SHA-256：`df25066d456d81883b5a7f97406901278b19bcbb7643a2059bc507e63ffbdc85`
- 共 123 个文件，解压后约 112.8 MiB；保留上游 Apache-2.0 LICENSE。

| 内容 | 相对于官方资产目录的路径 |
| --- | --- |
| 15 个 STL，含机身、腿部、顶盖和配件 | `hardware/printing/stl/` |
| STEP 装配模型 | `hardware/cad/Sesame-ESP32-v122.step` |
| Fusion 原始设计 | `hardware/cad/Sesame-ESP32-v122.f3z` |
| 固件、动作序列、表情与控制说明 | `firmware/` |
| 装配、接线与打印说明 | `docs/`、`hardware/printing/README.md` |
| 物料表与电路板文件 | `hardware/bom/`、`hardware/pcb/` |

完整逐文件 SHA-256 见 [reports/asset_manifest.json](reports/asset_manifest.json)。123 个文件均已验证哈希，15 个二进制 STL 的文件长度均与三角面数匹配；验证结果见 [reports/asset_validation.json](reports/asset_validation.json)。这只是文件完整性检查，不代表机器人动力学已经验证。

主仓库下载快照没有 URDF、USD、MJCF 或其他仿真 XML 模型；三个附带 ZIP 均为 PCB Gerber 数据。其 README 的 “Sesame ML Tools” 章节链接了另一个包含动力学模型的项目，已继续导入如下。

## Sesame ML 仿真资产

来源：[lukehollis/sesame-ml](https://github.com/lukehollis/sesame-ml)，是官方主仓库链接的独立实验项目，不是课程组发布的统一环境。本次只导入源文件和模型，没有安装它的 Python 包或执行其训练程序。

- 目录：[assets/upstream/sesame-ml](assets/upstream/sesame-ml)
- URDF：[sesame.urdf](assets/upstream/sesame-ml/src/sesame_ml/assets/urdf/sesame.urdf)
- MuJoCo MJCF：[sesame.xml](assets/upstream/sesame-ml/src/sesame_ml/assets/mjcf/sesame.xml)
- 相对路径所需的 11 个网格：`assets/upstream/sesame-ml/src/sesame_ml/assets/meshes/`
- 完整快照共 124 个文件，包含模型说明、任务与训练参考代码；许可与归属文件均保留。
- 版本：`a3910f67b4aeb9ca9080a1ba1ffa971e53e35a18`
- ZIP SHA-256：`f216ec2ffa43369a58de1e672c61573a5e53f915f6bedb06f62887f6655a7bbe`
- 逐文件来源与哈希：[reports/sesame_ml_manifest.json](reports/sesame_ml_manifest.json)

参考模型使用 `+x` 向前、`+y` 向左、`+z` 向上，URDF 长度为米、关节角为弧度。固件通道顺序为 `[R1, R2, L1, L2, R4, R3, L3, L4]`；仿真器实际返回的 DOF 顺序必须按关节名映射，不能假设与固件顺序相同。

静态检查已通过：15 个 link、8 个转动关节、6 个固定关节，单根连通关节树，11 个网格引用均存在；9 个有质量的刚体都有质量/惯量描述，声明总质量 0.384 kg。124 个文件和 ZIP 的哈希均与清单一致。详见 [reports/sesame_model_metadata.json](reports/sesame_model_metadata.json)。

模型作者注明质量和惯量为工程估算，默认标定文件是 `calibrated: false`；这些值尚未对课程真机标定。该项目的 Orange Pi/IMU/相机部署方案也不同于 PDF 的 ESP32 预测状态开环部署。上游 README 提到 NOTICE，但此快照没有该文件；已保留实际存在的 LICENSE 和源文件归属说明。

## 已导入 Isaac 的 USD

入口文件：[assets/isaac/sesame/sesame.usd](assets/isaac/sesame/sesame.usd)。移动或分享时应保留整个 `assets/isaac/sesame/` 目录，里面包含 USD 所需的网格/子层。

该文件由上述 URDF 转换，保留浮动基座和 8 个转动关节，合并 6 个无质量的固定标记 frame 后共有 9 个刚体。导入后的质量总和与源描述的 0.384 kg 一致。原 URDF 保留在上游快照中。

本机适配只发生在 [scripts/import_sesame_isaac.py](scripts/import_sesame_isaac.py) 和新生成文件中：适配 Isaac Sim 4.5 的 URDF 导入 API；在派生副本中把 STL 文件名的连字符替换为下划线，以绕过旧导入器的 USD prim 命名问题。网格内容哈希保持不变，源文件未改动。

导入检查使用上游 MJCF 的参考位置驱动刚度 5，以及 URDF 的阻尼 0.0025、力矩上限 0.18 N·m 和速度上限约 10.472 rad/s。这些是上游估算参数；自碰撞关闭，没有据此认定舵机或接触模型已标定。

GPU 检查：8 个关节在 `cuda:0` 上完成 32 步，每步 0.002 秒，关节与基座状态均有限。测试只覆盖短时悬空加载与参考角位置驱动，不覆盖脚底接触、站立稳定、行走或 PPO 训练。完整结果：[reports/sesame_isaac_import.json](reports/sesame_isaac_import.json)。

| 顺序 | 关节名称 |
| --- | --- |
| 固件/模型动作顺序 | `R1 R2 L1 L2 R4 R3 L3 L4` |
| 此次 Isaac DOF 顺序 | `L1 L2 R1 R2 L3 L4 R3 R4` |
| 从 Isaac 状态取固件顺序的下标 | `[2, 3, 0, 1, 7, 6, 4, 5]` |

重新转换并检查会覆盖派生 USD 和对应运行报告：

```bash
CONDA_NO_PLUGINS=true PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/miniforge3/bin/conda run -n env_isaaclab --no-capture-output \
  /home/ubuntu/miniforge3/envs/env_isaaclab/bin/python \
  /home/ywwang/sesame/scripts/import_sesame_isaac.py
```

## 现有 Isaac 环境

| 项目 | 检查结果 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 D，24 GB 级显存 |
| NVIDIA 驱动 | 570.133.07 |
| Python | `/home/ubuntu/miniforge3/envs/env_isaaclab/bin/python`，3.10.19 |
| Isaac Sim | `/home/ubuntu/isaacsim`，4.5.0-rc.36 |
| Isaac Lab | `/home/ubuntu/jrWork/IsaacLab`，VERSION 为 2.3.0，Python 包元数据为 0.49.0 |
| PyTorch | 2.7.0+cu128 |
| RL 库 | RSL-RL 3.1.2、skrl 1.4.3、RL-Games 1.6.1、Stable-Baselines3 2.7.1 |

已实测：

1. Isaac Lab 无界面启动成功。
2. 16 个刚体在 `cuda:0` 上推进 32 个物理步，状态张量位于 GPU，数值有限并响应重力。
3. 独立合成数据在 CUDA 上完成一次前向、反向传播及 Adam 参数更新。

这些是最初的通用环境检查；后续已增加接触验证并完成上文的 PPO 基线。全部模型和数值验证均在 GPU 上进行。

已观察到 Isaac 4.5 在测试通过后清理退出会等待。检查脚本先保存结果，再为自身进程设置 15 秒退出上限；报告明确记录 `shutdown.completed=false` 和 `forced_process_exit=true`。没有终止其他 GPU 进程。

详细版本和状态见 [reports/isaac_environment.json](reports/isaac_environment.json)，实际运行结果见 [reports/isaac_runtime_smoke.json](reports/isaac_runtime_smoke.json)，输出日志为 `reports/isaac_runtime_smoke.log`。

重跑检查：

```bash
cd /home/ywwang/sesame
CONDA_NO_PLUGINS=true PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/miniforge3/bin/conda run -n env_isaaclab --no-capture-output \
  /home/ubuntu/miniforge3/envs/env_isaaclab/bin/python \
  /home/ywwang/sesame/scripts/check_isaac_runtime.py
```

需要 Conda 激活钩子载入 Isaac Sim 的运行路径；直接调用未激活的 Python 会找不到 `isaacsim`。环境检查没有安装或升级任何包。

## 后续对齐课程与真机

当前已建立本地 Isaac 训练环境。仍需获取课程组统一机器人模型与接口，避免训练模型的关节映射和评分环境不一致：

1. 将已取得的 8 关节参考模型与课程指定版本对齐，核对单位、装配坐标、关节轴向、顺序及真机标定。
2. 核对参考模型的质量、惯量、碰撞几何、足端接触/摩擦、舵机力矩/速度/驱动增益及站立姿态；公开模型参数不等于课程真机实测参数。
3. 对齐课程动作缩放、限位、限速、控制频率、观测归一化、终止条件与统一评测接口，再接入 PPO。

已完成的 PPO 基线使用独立保存的本地奖励与评测配置，没有将公开项目接口当作课程统一接口。
