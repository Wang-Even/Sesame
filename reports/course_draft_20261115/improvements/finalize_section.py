#!/usr/bin/env python3
"""Add measured conclusions to the generated report without changing run sources."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
summary = json.loads((OUT / 'summary.json').read_text())
rows = summary['results']


def row(model, condition, speed):
    return next(r for r in rows if (r['model'], r['condition'], r['target_speed_m_s']) == (model, condition, speed))


def relative_change(new, old, key):
    return 100 * (new[key] / old[key] - 1)


target = OUT / 'section_improvements.md'
generated = OUT / 'section_improvements_generated.md'
if not generated.exists():
    generated.write_bytes(target.read_bytes())
text = generated.read_text()
for previous, current in [('标准/随机条件', '固定参数/域随机化条件'),
                          ('标准参数', '固定参数'), ('随机参数', '域随机化'),
                          ('标准条件', '固定参数条件'), ('随机条件', '域随机化条件')]:
    text = text.replace(previous, current)
bn, br = row('baseline', 'nominal', .1), row('baseline', 'randomized', .1)
sn, sr = row('smooth', 'nominal', .1), row('smooth', 'randomized', .1)
smooth = (
    f"本轮动作差惩罚加倍后，固定参数/域随机化条件下的归一化动作变化率分别下降 "
    f"{-relative_change(sn,bn,'normalized_action_rate_rms_per_s'):.2f}% / "
    f"{-relative_change(sr,br,'normalized_action_rate_rms_per_s'):.2f}%。"
    f"但实际关节加速度只在域随机化条件下下降 {-relative_change(sr,br,'actual_joint_acceleration_rms_rad_s2'):.2f}%，"
    f"固定参数条件反而增加 {relative_change(sn,bn,'actual_joint_acceleration_rms_rad_s2'):.2f}%。"
    f"域随机化条件有效速度由 {br['mean_forward_speed_m_s']:.4f} 降至 {sr['mean_forward_speed_m_s']:.4f} m/s，"
    f"更接近 0.10 m/s，速度 RMSE 和航向误差也下降；固定参数条件速度则变为 {sn['mean_forward_speed_m_s']:.4f} m/s，超过目标更多。"
    "因此，这次改动只能说明动作信号变化略有减少，尚未证明实际关节运动在两种条件下都更平滑。"
    "后续可尝试小权重的实际关节加速度或机械功率惩罚，并继续同时检查速度、航向与至少三个训练种子的结果。\n\n"
)
text = text.replace('动作变化率为相邻归一化动作差的 RMS', smooth + '动作变化率为相邻归一化动作差的 RMS', 1)
fn, fr = row('fast', 'nominal', .2), row('fast', 'randomized', .2)
fast = (
    "低速策略直接接收 0.20 m/s 指令时几乎不前进，说明固定速度训练不能保证对未见速度指令的泛化。"
    f"高速组重新训练后，固定参数/域随机化条件有效速度达到 {fn['mean_forward_speed_m_s']:.4f} / {fr['mean_forward_speed_m_s']:.4f} m/s，"
    f"但相对目标仍分别超出 {(fn['mean_forward_speed_m_s']/.2-1)*100:.1f}%、低于 {(1-fr['mean_forward_speed_m_s']/.2)*100:.1f}%。"
    f"高速组的 256 回合测试未出现跌倒或提前终止，航向 RMS 为 {fn['heading_rms_deg']:.2f}° / {fr['heading_rms_deg']:.2f}°；"
    "这些结果支持本轮测试中能够较快前进，但不等于严格无偏航或速度已精确跟踪。"
    "下一步可在训练中随机采样 0.10–0.20 m/s 速度指令并逐步提高上限，再检查固定参数下的超速、域随机化下的欠速及侧移。\n\n"
    f"高速组固定参数条件下，净位移速度 / 50 Hz 采样 vx 均值为 {fn['mean_forward_speed_m_s']:.4f} / {fn['sampled_mean_vx_m_s']:.4f} m/s；"
    f"域随机化条件下为 {fr['mean_forward_speed_m_s']:.4f} / {fr['sampled_mean_vx_m_s']:.4f} m/s。"
    "固定参数条件两者接近，超速现象不能用这两个均值的差异解释；域随机化条件的统计差异更明显，但目前不能据此确定根因。"
    "后续可对照使用控制区间位移 / Δt 计算速度，使奖励与评估的速度口径一致，并检查接触运动与采样时刻的影响。\n\n"
)
text = text.replace('前进速度为 X 方向净位移除以回合时长', fast + '前进速度为 X 方向净位移除以回合时长', 1)
text = text.replace('完整训练进程（分钟）', '程序启动至训练结束（分钟，关闭前）')
text = text.replace('| baseline |', '| 原设置 PPO |').replace('| smooth |', '| 动作差惩罚加倍 |').replace('| fast |', '| 0.20 m/s 训练 |')
text = text.replace('结果目录：',
    '以上为本轮三组串行实测时间；第 2 节为历史运行记录，运行负载与计时环境不同，不能把跨轮用时差异归因于奖励修改。本表也不混入第 6 节另一任务配置下的 PPO-Sim 用时。程序时间计至训练结果保存、仿真关闭之前，不包含之后的统一评估。\n\n结果目录：', 1)
target.write_text(text)
(OUT / 'editorial_provenance.json').write_text(json.dumps({
    'generated_section_sha256': hashlib.sha256(generated.read_bytes()).hexdigest(),
    'final_section_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'note': 'Post-run report interpretation and timing-label clarification only. Training/evaluation/plot source hashes remain unchanged.'
}, indent=2) + '\n')
print(target)
