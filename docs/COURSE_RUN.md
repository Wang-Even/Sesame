# 课程实验的运行与权重

以下命令从 Sesame 仓库根目录运行。先执行 `git lfs pull` 获取机器人网格、模型权重及视频。Isaac 速度指令行走和 MuJoCo 参考轨迹跟踪分别使用独立的仿真环境。

## MuJoCo 参考轨迹跟踪

训练使用 SuperMimic `5efbf114`、mjlab research `dd51a9fc` 和 Sesame 任务适配代码。先从公开上游恢复这两个快照。脚本依照 `third_party/*.provenance.json` 核验每个文件的 SHA-256，保留各自的 LICENSE。

```bash
python3 scripts/restore_course_dependencies.py
```

离线机器也可运行 `python3 scripts/restore_course_dependencies.py --archive-dir /path/to/archives`。该目录应包含两个对应提交的 GitHub 源码 tar.gz，文件名分别为 `supermimic_official_5efbf114.tar.gz` 和 `mjlab_research_dd51a9fc.tar.gz`。

本实验采用 Linux x86_64、NVIDIA GPU、Python 3.13.18、PyTorch 2.14.1+cu126、MuJoCo 3.16.0、MuJoCo Warp 3.16.0 和 Warp 1.19.0+cu12。依赖清单来自实验环境，mjlab 使用恢复后的本地源码。启动器通过 `.venv_supermimic_research/bin/python` 调用训练。

```bash
uv venv --python 3.13.18 .venv_supermimic_research
uv pip install --python .venv_supermimic_research/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match -r configs/course_mujoco_requirements.txt
```

`datasets/course_comparison_20261115` 包含原始、DeepSeek、Codex 三种来源的参考数据及划分。每种来源使用 24 条训练、4 条验证和 4 条测试轨迹。训练已包含在数据集中的 Agent 参考时，无需调用模型 API。

以下命令依次执行四组主实验、三组消融、延长预算的 PPO，并评估最终运动表现。指定三个新的输出目录，重新运行时更换目录名。命令末尾加 `--check-config` 可先检查配置、数据及上游文件。

```bash
.venv_supermimic_research/bin/python scripts/run_course_dense_rerun.py \
  --dataset datasets/course_comparison_20261115 \
  --suite runs/my_course_suite --extended runs/my_extended_ppo \
  --motion runs/my_motion_results
```

报告使用训练预算结束时的 `final.pt`。下表前七项路径均相对 `runs/course_comparison_dense_20261116/`。

| 方法 | 权重路径 |
|---|---|
| PPO | `ppo/final.pt` |
| SuperMimic（原始参考） | `supermimic/final.pt` |
| SuperMimic（DeepSeek 参考） | `supermimic_deepseek/final.pt` |
| SuperMimic（Codex 参考） | `supermimic_codex/final.pt` |
| 想象长度 16 | `supermimic_h16/final.pt` |
| 想象长度 64 | `supermimic_h64/final.pt` |
| 移除 Critic | `supermimic_no_critic/final.pt` |

延长预算 PPO 的权重位于 `runs/course_ppo_extended_dense_20261116/final.pt`。使用相同的 12 条保留测试轨迹评估四组主实验时，执行下列命令。可先加 `--check-config` 进行文件检查。

```bash
.venv_supermimic_research/bin/python scripts/evaluate_course_motion.py \
  --suite runs/course_comparison_dense_20261116 \
  --dataset datasets/course_comparison_20261115 --output runs/my_eval
```

## Isaac 四足行走基线

Isaac 基线采用 Isaac Sim 4.5、Isaac Lab 2.3 和 PyTorch 2.7。PPO 的训练及评估入口为 `scripts/train_heading_ppo.py` 和 `scripts/evaluate_heading_ppo.py`。SAC 和 DQN 共用 `scripts/train_offpolicy.py` 和 `scripts/evaluate_offpolicy.py`。动作平滑性对照使用 `scripts/train_ppo_motion_study.py` 和 `scripts/evaluate_ppo_motion_study.py`。

| 方法 | 权重路径，种子可取 42、43、44 |
|---|---|
| PPO | `runs/ppo_heading_20261105/training/heading_seed{seed}/final.pt` |
| SAC | `runs/offpolicy_heading_20261107/training/sac_seed{seed}/final.pt` |
| DQN | `runs/offpolicy_heading_20261107/training/dqn_seed{seed}/final.pt` |

平滑性实验使用 `runs/ppo_motion_study_20261115/training/baseline_seed42/final.pt` 和同目录层级的 `smooth_seed42/final.pt`。实验结果及评估指标见[课程报告](../reports/course_rerun_20261116/report.pdf)。
