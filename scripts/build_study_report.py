#!/usr/bin/env python3
"""Build a Chinese report from saved scalar metadata, with no model or simulation imports."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/robustness_ablation_20261103"
OUT = ROOT / "reports/ppo_study_summary_20261103"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font_manager.fontManager.addfont(FONT)
font_manager.fontManager.addfont(BOLD)
FP = font_manager.FontProperties(fname=FONT)
FB = font_manager.FontProperties(fname=BOLD)
plt.rcParams.update({"font.family": FP.get_name(), "pdf.fonttype": 3, "axes.unicode_minus": False})
INK, MUTED, BLUE = "#172c45", "#53677f", "#245ee9"
NAMES = {"baseline": "原 PPO 基线", "full_dr": "完整域随机化", "fixed_delay": "固定时延 DR"}
ORDER = list(NAMES)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def wrap(s, width=114):
    lines = []
    for paragraph in s.split("\n"):
        row, n = "", 0
        for c in paragraph:
            w = 2 if unicodedata.east_asian_width(c) in "WF" else 1
            if n + w > width:
                lines.append(row)
                row, n = "", 0
            row += c
            n += w
        lines.append(row)
    return "\n".join(lines)


def txt(fig, x, y, s, size=12, color=INK, bold=False, width=None):
    return fig.text(x, y, wrap(s, width) if width else s, ha="left", va="top",
                    fontsize=size, color=color, fontproperties=FB if bold else FP,
                    linespacing=1.6)


def page(title, num):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    txt(fig, .055, .946, title, 23, bold=True)
    txt(fig, .055, .885, "SESAME · PPO 行走实验 | 2026-11-03", 10, MUTED)
    footer(fig, num)
    return fig


def footer(fig, num):
    txt(fig, .055, .039, "本地参考仿真 · 每组 3 个训练种子 · 误差条为样本标准差", 8.5, MUTED)
    fig.text(.945, .039, str(num), fontsize=9, color=MUTED, ha="right", va="top")


def block(fig, y, title, body, size=12, width=115):
    txt(fig, .06, y, title, 14, bold=True)
    txt(fig, .06, y-.047, body, size, width=width)


def pm(group, metric, digits=5):
    d = group[metric]
    return f"{d['mean']:.{digits}f} ± {d['sd_across_training_seeds']:.{digits}f}"


def make_table(fig, bbox, headings, rows, widths=None, size=11):
    ax = fig.add_axes(bbox)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headings, loc="center", cellLoc="center",
                     colWidths=widths, bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(size)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dce4ef")
        cell.get_text().set_fontproperties(FB if r == 0 else FP)
        cell.get_text().set_fontsize(size)
        cell.get_text().set_color(INK)
        cell.set_facecolor("#edf3fc" if r == 0 else ("#f8fafc" if r % 2 == 0 else "white"))
    return table


def html_table(headings, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(x))}</th>" for x in headings) + \
           "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table>"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fm = read(OUT / "figures_manifest.json")
    editorial = read(OUT / "editorial_review.json")
    summary = read(STUDY / "analysis/summary.json")
    protocol = read(STUDY / "protocol.json")
    old_result = read(STUDY / "result.json")
    # Verify frozen inputs and previous results before writing any report artifacts.
    checked = {}
    for p, expected in editorial["source_inputs_sha256"].items():
        actual = sha(Path(p))
        assert actual == expected, f"Changed source: {p}"
        checked[p] = actual
    for rel, meta in old_result["artifacts"].items():
        assert sha(ROOT / rel) == meta["sha256"], f"Changed old result: {rel}"
        checked[str(ROOT / rel)] = meta["sha256"]
    for item in fm["figures"]:
        for meta in item["artifacts"].values():
            assert sha(OUT / meta["path"]) == meta["sha256"]
    for item in fm["tables"]:
        assert sha(OUT / item["path"]) == item["sha256"]
    for dirname in ["data", "media"]:
        (OUT / dirname).mkdir(exist_ok=True)
    for source, target in [(STUDY / "analysis/summary.json", "data/evaluation_summary.json"),
                           (STUDY / "protocol.json", "data/protocol.json"),
                           (ROOT / "runs/ppo_dr_20261102_seed42/ppo_config.json", "data/ppo_config.json"),
                           (STUDY / "videos/ppo_study_comparison_seed3042.mp4", "media/comparison.mp4"),
                           (STUDY / "videos/comparison_thumbnail.png", "media/comparison_thumbnail.png"),
                           (STUDY / "videos/comparison_validation.json", "data/video_validation.json")]:
        shutil.copyfile(source, OUT / target)
        checked[str(source)] = sha(source)

    g = summary["groups"]
    metric = "velocity_x_rmse_m_s"
    improvement = 100 * (1-g["randomized"]["full_dr"][metric]["mean"]/g["randomized"]["baseline"][metric]["mean"])
    ablation = 100 * (1-g["randomized"]["full_dr"][metric]["mean"]/g["randomized"]["fixed_delay"][metric]["mean"])
    cost = 100 * (g["nominal"]["full_dr"][metric]["mean"]/g["nominal"]["baseline"][metric]["mean"]-1)
    primary_rows = [[NAMES[v], pm(g["nominal"][v], metric), pm(g["randomized"][v], metric)] for v in ORDER]
    caps = editorial["captions_zh"]
    narratives = [
        caps["training_timing"] + "\n" + caps["training_vs_evaluation"],
        caps["randomized_benefit"] + "\n" + caps["nominal_tradeoff"],
        caps["delay_ablation"] + "\n消融同时固定响应时间常数与指令延迟，不能分别判断二者的贡献。",
        caps["distance"] + "\n随机条件下完整 DR 的平均绝对终点偏航为 13.10°，横向位移为 0.0354 m；这不等于已消除侧行或提高机械能效。",
        caps["smoothness_result"] + "\n" + caps["falls"],
    ]
    final_return_rows = []
    for v in ORDER:
        last = editorial["training_milestone_statistics"][v][-1]["mean_episode_return"]
        final_return_rows.append([NAMES[v], f"{last['mean']:.3f} ± {last['sample_sd_across_training_seeds']:.3f}"])

    with PdfPages(OUT / "report.pdf", metadata={"Title": "Sesame PPO 训练曲线与对比报告", "Author": "Sesame experiment report", "Subject": "Saved training and evaluation metadata"}) as pdf:
        fig = page("Sesame PPO：训练曲线与方法对比", 1)
        txt(fig, .06, .815, "主要结果：随机动力学下速度跟踪更准，同时存在标称性能取舍。", 17, bold=True)
        cards = [(f"−{improvement:.1f}%", "随机条件 RMSE / 相对原 PPO", BLUE),
                 (f"−{ablation:.1f}%", "随机条件 RMSE / 相对固定时延", "#168761"),
                 (f"+{cost:.1f}%", "标称条件 RMSE / 相对原 PPO", "#cd6715")]
        for i, (value, label, color) in enumerate(cards):
            x = .06 + i*.30
            fig.add_artist(FancyBboxPatch((x,.605), .28,.155, boxstyle="round,pad=0.008", transform=fig.transFigure, facecolor="#f3f6fb", edgecolor="none"))
            txt(fig,x+.017,.738,value,27,color,True)
            txt(fig,x+.017,.656,label,10.5,MUTED)
        txt(fig, .06, .56, "前向速度 RMSE（m/s，越低越好）：均值 ± 训练种子样本标准差", 12, bold=True)
        make_table(fig, [.06,.343,.88,.18], ["方法", "标称条件", "随机条件"], primary_rows, [.28,.36,.36], 12)
        txt(fig, .06, .293, "实验规模", 13, bold=True)
        txt(fig, .06, .25, "3 组 × 3 个训练种子；每策略 250 次更新 / 2.048 百万环境步。\n统一评测 2 个条件、每策略每条件 128 回合，共 2304 回合。", 12)
        txt(fig, .06, .15, "结论边界", 13, bold=True)
        txt(fig, .06, .111, "未证明整体减少抖动；全部评测零跌倒，无法比较跌倒率改善。\n结果限于本地参考仿真和训练参数范围内，不代表官方成绩或实机性能。", 11.2)
        fig.savefig(OUT / "overview.png", dpi=160)
        pdf.savefig(fig)
        plt.close(fig)

        for i, item in enumerate(fm["figures"], 2):
            fig = plt.figure(figsize=(11.69,8.27), facecolor="white")
            ax=fig.add_axes([.015,.06,.97,.925])
            ax.imshow(plt.imread(OUT/item["artifacts"]["png"]["path"]))
            ax.axis("off")
            footer(fig,i)
            pdf.savefig(fig,dpi=200)
            plt.close(fig)

        fig = page("实验设置与统计口径",7)
        block(fig,.811,"任务与三组方法", "平坦硬地，目标世界 x 速度 0.10 m/s，10 秒/回合，8 个动作维度；控制 50 Hz，物理步长 0.002 s。三组使用相同 PPO、网络、奖励与预算。", width=113)
        rows = [[NAMES["baseline"], "原固定物理参数、原初始化和控制器"],
                [NAMES["full_dr"], "物理参数、执行器响应/延迟和初态随机化"],
                [NAMES["fixed_delay"], "其余同完整 DR；响应 τ=0.04 s，指令延迟=0.02 s"]]
        make_table(fig,[.06,.505,.88,.145],["组别","训练设置"],rows,[.23,.77],11)
        block(fig,.469,"PPO 与训练预算", "256 环境 × 32 步 rollout；250 次更新；种子 42/43/44。网络 128–128–64（ELU），学习率 0.0003（自适应 KL=0.01），clip=0.2，γ=0.99，λ=0.95；5 epochs、4 minibatches，熵系数 0.005。原 PPO 与完整 DR 的种子 42 复用既有等预算模型，其余 7 个新训练。",size=11.4,width=118)
        block(fig,.265,"统一评测与误差条", "全部策略使用同一环境，以最终检查点的确定性动作评测。测试种子 3042/4042，各取 64 回合；初态及实际物理参数严格配对。先计算每回合的 500 步瞬时速度 RMSE，再平均 128 回合，最后汇总 3 个训练种子的均值 ± 样本标准差；误差条不是置信区间。",size=11.4,width=118)
        txt(fig,.06,.100,"标称条件恢复原控制器零附加滞后/延迟；随机测试使用训练参数范围内的新种子。",10.8,MUTED)
        pdf.savefig(fig)
        plt.close(fig)

        fig=page("如何解读与使用这些结果",8)
        block(fig,.811,"训练曲线说明学习进展，统一测试决定性能比较",caps["training_timing"]+" 末 10 次更新的回报见下表；训练回报已包含最近 100 个已完成回合平均，再作 10 次更新平滑。",size=11.6,width=118)
        make_table(fig,[.06,.543,.88,.135],["方法","末 10 次更新的训练回报"],final_return_rows,[.40,.60],11)
        block(fig,.506,"平滑性指标：50 Hz 采样的运动代理量", "目标变化率 = RMS(Δq_target/0.02)，目标加速度 = RMS(Δ²q_target/0.02²)；实际关节加速度 = RMS(Δv_joint/0.02)，v_joint 为关节角速度。目标为执行器最终施加目标，差分排除自动重置跳变。指标包含正常步态，不是抖动百分比、高频功率或 jerk；动作限幅比例单独报告。",size=11.2,width=120)
        block(fig,.313,"实验能支持的结论", "完整 DR 对随机条件速度跟踪的改善在 3 个种子方向一致；相对固定时延组的额外收益主要来自种子 42/43，种子 44 近乎持平。基线与完整 DR 比较包含初态和平均执行器动力学变化，不能归因于单一因素。仅 3 个训练种子，未宣称统计显著性或普遍优势。",size=11.2,width=120)
        block(fig,.122,"附件", "5 张 PNG 与矢量 PDF、4 份中文 CSV、HTML 报告、统计/协议 JSON 及 10 秒三组对照视频。视频仅用于演示，不能代替多种子统计。",size=10.8,width=124)
        pdf.savefig(fig)
        plt.close(fig)

    sections=[]
    for item, text in zip(fm["figures"], narratives):
        png=OUT/item["artifacts"]["png"]["path"]
        uri="data:image/png;base64,"+base64.b64encode(png.read_bytes()).decode()
        paragraphs="".join(f"<p>{html.escape(s)}</p>" for s in text.split("\n"))
        sections.append(f'<section id="{item["id"]}"><h2>{item["title"]}</h2>{paragraphs}<img src="{uri}" alt="{html.escape(item["title"])}"><p class="caption">{html.escape(item["caption"])}</p><p><a href="{item["artifacts"]["pdf"]["path"]}">下载矢量 PDF</a> · <a href="{item["artifacts"]["png"]["path"]}">下载 PNG</a></p></section>')
    filelist="".join(f'<li><a href="{t["path"]}">{Path(t["path"]).name}</a>：{html.escape(t["caption"])}</li>' for t in fm["tables"])
    method_rows=[["训练重复", "每组种子 42/43/44；256×32×250 = 2,048,000 环境转移/策略"],
                 ["测试预算", "3 组×3 模型×2 条件×128 回合 = 2304 回合；每回合 10 s"],
                 ["测试配对", "测试种子 3042/4042；所有策略的初态、关节状态及实际物理域参数严格一致"],
                 ["主指标", "每回合 RMSE = sqrt(mean_t[(v_x(t)−0.10)²])，每回合 500 个 50 Hz 控制样本"],
                 ["统计", "先平均每策略 128 回合，再汇总 3 个训练种子的均值 ± 样本 SD，非置信区间"],
                 ["固定时延消融", "与完整 DR 相同，仅固定响应 τ=0.04 s、指令延迟=1 控制步（0.02 s）"],
                 ["PPO", "lr=3e−4，自适应 KL=0.01；clip=0.2，γ=0.99，λ=0.95，5 epochs×4 minibatches；entropy=0.005"],
                 ["网络", "Actor/Critic 均为 128–128–64，ELU；观测归一化；初始动作标准差 0.5"],
                 ["奖励", "三组使用同一奖励：速度跟踪/前进/直立奖励，横移、转动、垂直运动、扭矩、动作变化及姿态偏差惩罚；未为本实验修改"],
                 ["标称与随机条件", "标称条件恢复原零附加滞后/延迟；随机条件采样训练范围内的新动力学，不是范围外测试"]]
    css="""body{margin:0;background:#f2f5f9;color:#172c45;font:16px/1.8 system-ui,'Noto Sans CJK SC',sans-serif}main{max-width:1100px;margin:auto;padding:36px 22px}header,section{background:white;padding:28px 34px;margin-bottom:24px;border-radius:12px}h1{font-size:32px;margin:0 0 12px}h2{font-size:24px;margin-top:0}h3{font-size:19px}p{margin:12px 0}a{color:#245ee9}img{display:block;width:100%;height:auto}.caption,.muted{color:#53677f;font-size:14px}table{border-collapse:collapse;width:100%;margin:18px 0;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;border:1px solid #dce4ef;text-align:left}th{background:#edf3fc}tr:nth-child(even){background:#f8fafc}.cards{display:flex;gap:14px}.card{flex:1;background:#f3f6fb;padding:16px;border-radius:8px}.value{font-size:30px;font-weight:700;color:#245ee9}.card small{display:block}video{width:100%;background:#172c45}li{margin:8px 0}@media(max-width:700px){header,section{padding:20px 14px}.cards{display:block}.card{margin:10px 0}table{font-size:13px}h1{font-size:25px}}@media print{body{background:white}main{max-width:none;padding:0}header,section{break-inside:avoid;border-radius:0;padding:10px}video{display:none}}"""
    content=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sesame PPO 训练曲线与方法对比</title><style>{css}</style></head><body><main>
<header><p class="muted">SESAME · 2026-11-03 · 已保存训练与评测数据汇总</p><h1>Sesame PPO：训练曲线与方法对比</h1><p>完整域随机化改善了随机动力学下的速度跟踪，同时存在标称性能取舍；本轮未证明整体减少抖动或降低跌倒率。</p><div class="cards"><div class="card"><span class="value">−{improvement:.1f}%</span><small>随机条件 RMSE / 相对原 PPO</small></div><div class="card"><span class="value">−{ablation:.1f}%</span><small>随机条件 RMSE / 相对固定时延</small></div><div class="card"><span class="value">+{cost:.1f}%</span><small>标称条件 RMSE / 相对原 PPO</small></div></div>
{html_table(["方法","标称 RMSE（m/s）","随机 RMSE（m/s）"],primary_rows)}<p class="caption">均值 ± 三个训练种子的样本标准差；越低越好。每个模型/条件先平均 128 个评测回合。</p><p><a href="report.pdf">下载 8 页报告 PDF</a> · <a href="#attachments">表格与视频</a></p></header>
{''.join(sections)}
<section><h2>实验设置与指标口径</h2>{html_table(["项目","设置"],method_rows)}<h3>末段训练回报</h3>{html_table(["方法","末 10 次更新均值 ± 种子样本 SD"],final_return_rows)}<p>{html.escape(caps["training_return"])}</p><h3>平滑性计算</h3><p>50 Hz 控制采样，Δt=0.02 s；对 8 个关节与时间计算 RMS。目标变化率为 Δq_target/Δt，目标加速度为 Δ²q_target/Δt²，实际关节加速度为 Δv_joint/Δt，v_joint 表示关节角速度。q_target 指经过限幅、速率约束、延迟与响应滞后后的最终目标角。每回合先计算 RMS，再平均回合、汇总种子；排除自动重置跳变。</p><p>目标二阶差分使用 499 个有效间隔；实际关节加速度包括从已知重置关节速度到第一控制步的差分，共 500 个样本。原始动作限幅比例为裁剪前 |a|&gt;1 的分量占比。</p><p>{html.escape(caps["smoothness"])}</p><h3>结论边界</h3><p>仅 3 个训练种子，未进行预先指定的显著性检验。基线与完整 DR 比较包含初态与平均执行器动力学的变化；完整 DR 与固定时延比较同时改变响应滞后和延迟的变化范围，不能区分二者各自贡献。两份种子 42 历史模型按协议复用，开发评测种子 2042 未纳入正式测试。</p><p>{html.escape(caps["scope"])}</p></section>
<section id="attachments"><h2>表格、视频与溯源</h2><ul>{filelist}</ul><p>CSV 为 UTF-8 BOM，可直接用 Excel 打开；全部图表同时提供 PNG 和矢量 PDF。</p><video controls preload="metadata" poster="media/comparison_thumbnail.png"><source src="media/comparison.mp4" type="video/mp4"></video><p><a href="media/comparison.mp4">下载 10 秒三组并排视频</a>。固定训练种子 42、测试种子 3042，随机条件，配对初态与物理参数；仅作单回合示例，不代表组均值。单环境与批量环境的抽样形状不同。</p><p><a href="data/evaluation_summary.json">完整统计 JSON</a> · <a href="data/protocol.json">实验协议</a> · <a href="data/ppo_config.json">PPO 配置</a> · <a href="report_manifest.json">来源与文件哈希</a></p><p class="caption">本 HTML 的图表已嵌入；视频、CSV 与下载链接需保留解压后的目录结构。</p></section></main></body></html>'''
    (OUT/"report.html").write_text(content)
    (OUT/"README.txt").write_text("Sesame PPO 训练曲线与方法对比（2026-11-03）\n\n优先阅读 report.pdf（8 页）或 report.html（图表已内嵌）。\nfigures/：5 张中文图，每张 PNG + 矢量 PDF。\ntables/：4 份 UTF-8 BOM CSV，包含组统计、逐种子指标与训练里程碑。\nmedia/：10 秒三组对照视频与缩略图，视频不替代统计。\ndata/：原始统计 JSON 副本、协议、PPO 配置与视频验证。\nreport_manifest.json：来源与全部产物 SHA-256。\n\n重建仅处理已保存元数据，不启动训练或仿真：\n  python3 /home/ywwang/sesame/scripts/plot_study_report.py\n  python3 /home/ywwang/sesame/scripts/build_study_report.py\n\n统计单位为每组 3 个独立训练种子，误差条为样本标准差。\n结果是本地参考仿真中的范围内泛化，不是官方评分或实机结论。\n")
    for p in [Path(__file__),ROOT/"scripts/plot_study_report.py",OUT/"figures_manifest.json",OUT/"editorial_review.json"]:
        checked[str(p)] = sha(p)
    artifacts={str(p.relative_to(OUT)):{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "report_manifest.json" and not any(part.startswith(".") for part in p.relative_to(OUT).parts)}
    write_json(OUT/"report_manifest.json",{
        "status":"complete", "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"Saved metadata report generation only; no training, inference, simulation or Git operations",
        "report_pdf_pages":8,"official_score":False,"independent_training_seeds_per_method":3,
        "training_runs":9,"evaluation_episodes":2304,
        "relative_rmse_changes_percent":{"randomized_full_dr_vs_baseline":-improvement,"randomized_full_dr_vs_fixed_delay":-ablation,"nominal_full_dr_vs_baseline":cost},
        "source_inputs_sha256":checked,"previous_study_artifact_hashes_verified":True,
        "artifacts":artifacts,
    })
    archive=OUT.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(OUT).parts):
                z.write(p, str(Path(OUT.name)/p.relative_to(OUT)))
    archive.with_suffix(".zip.sha256").write_text(f"{sha(archive)}  {archive.name}\n")
    print(json.dumps({"report":str(OUT/"report.pdf"),"archive":str(archive),"files":len(artifacts)+1,"zip_bytes":archive.stat().st_size,"verified_source_files":len(checked)},ensure_ascii=False))


if __name__ == "__main__":
    main()
