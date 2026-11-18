"""Summarize saved DeepSeek generation/validation records (no model computation)."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ROOT / 'runs/deepseek_report_pilot_20261115',
    ROOT / 'runs/deepseek_report_20261115',
    ROOT / 'runs/deepseek_report_seeded_20261115',
]
OUT = ROOT / 'reports/course_draft_20261115/deepseek'


def read(path):
    return json.loads(path.read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    history = []
    for run in RUNS:
        status = read(run / 'status.json')
        records = [read(p) for p in run.glob('candidates/**/*_api.json')]
        history.append(dict(
            run=str(run.relative_to(ROOT)), status=status['status'],
            accepted_count=status['accepted_count'], attempts=status['attempts'],
            attempted_api_calls=status['api_calls'], completed_api_records=len(records),
            elapsed_s=status.get('elapsed_s', status.get('finished_unix', status['started_unix']) - status['started_unix']),
            prompt_tokens=sum(r.get('usage', {}).get('prompt_tokens', 0) for r in records),
            completion_tokens=sum(r.get('usage', {}).get('completion_tokens', 0) for r in records),
            response_models=sorted({r['response_model'] for r in records}),
        ))
    formal = RUNS[-1]
    status = read(formal / 'status.json')
    if status['status'] != 'complete' or status['accepted_count'] != 32:
        raise ValueError('Formal collection is not complete with 32 admitted references')
    manifest = read(formal / 'accepted/manifest.json')
    codex_hashes = {
        read(p)['trajectory_sha256']: str(p.relative_to(ROOT))
        for p in (ROOT / 'runs/agent32_collection_20261114').glob('candidates/*/attempt_*/report.json')
        if 'trajectory_sha256' in read(p)
    }
    rows, duplicates, hashes = [], [], set()
    for entry in manifest['entries']:
        report = read(Path(entry['validation']))
        assert report['accepted'] and report['validation_complete']
        trials = [report['nominal'], *report['randomized']]
        assert len(trials) == 4 and all(t['passed'] for t in trials)
        h = report['trajectory_sha256']
        assert h not in hashes
        hashes.add(h)
        if h in codex_hashes:
            duplicates.append({'id': entry['id'], 'codex_report': codex_hashes[h]})
        for trial_index, trial in enumerate(trials):
            m = trial['metrics']
            rows.append(dict(id=entry['id'], trial=trial_index,
                condition='nominal' if trial_index == 0 else 'randomized',
                passed=trial['passed'], completed=m['completed'], fell=m['fell'],
                mean_forward_speed_m_s=m['mean_forward_speed_m_s'],
                heading_rms_deg=m['heading_rms_deg'],
                lateral_displacement_max_m=m['lateral_displacement_max_m'],
                root_tracking_rmse_m=m['root_tracking_rmse_m'],
                joint_tracking_rmse_rad=m['joint_tracking_rmse_rad'],
                report=entry['validation'], trajectory_sha256=h))
    assert not duplicates, duplicates
    with (OUT / 'accepted_trial_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = dict(
        formal_collection=str(formal.relative_to(ROOT)), history=history,
        dataset='Primary paired-experiment script will create whole-trajectory 24/4/4 splits from this collection',
        accepted_unique_references=len(hashes), accepted_physics_trials=len(rows),
        cross_source_exact_reference_duplicates=duplicates,
        reference_hash_comparison_scope='all 45 original Codex candidates, including 41 that pass current gates',
        caveat='Exact reference deduplication only; all gaits share calibrated engineering priors and are correlated. Parent experiment separately checks actual-state train/validation/test content hashes.',
        accepted_metric_ranges={k:{'min':min(r[k] for r in rows), 'mean':sum(r[k] for r in rows)/len(rows), 'max':max(r[k] for r in rows)}
            for k in ['mean_forward_speed_m_s','heading_rms_deg','lateral_displacement_max_m','root_tracking_rmse_m','joint_tracking_rmse_rad']},
        data_preparation_elapsed_s=sum(h['elapsed_s'] for h in history),
        attempted_api_calls=sum(h['attempted_api_calls'] for h in history),
        completed_api_records=sum(h['completed_api_records'] for h in history),
        prompt_tokens=sum(h['prompt_tokens'] for h in history),
        completion_tokens=sum(h['completion_tokens'] for h in history),
        protocol_sha256=hashlib.sha256((formal/'protocol.json').read_bytes()).hexdigest(),
        loader_verification=read(formal/'loader_verification.json'),
        positive_example_provenance='DeepSeek pilot candidate_000 API output; its exact reference matches historical Codex candidate_001, which belongs to the Codex TRAIN split, not validation/test. Formal final 32 contain no such exact cross-source duplicate.',
    )
    (OUT / 'collection_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    lo, hi = (summary['accepted_metric_ranges']['mean_forward_speed_m_s'][k] for k in ('min','max'))
    text = f'''## DeepSeek 参考数据补充实验

使用 `deepseek-flash`，流程为规划、完整代码生成、运动学检查、真实仿真检验，以及失败后的诊断与修订。最终程序全部来自已保存的真实 API 回复。工程上下文包含机器人几何、与 Codex 组相同的已校准反馈控制器，以及 DeepSeek 自己在试采中生成并通过检验的一条正例；因此这是共享工程先验的参考数据生成实验，不是无指导生成或语言模型能力盲测。

最终采集得到 **32 条不同参考**，每条 6 秒、50 Hz。每条均通过正常条件和 3 次域随机化仿真，共 **128 次入库前仿真检验**。与 Codex 历史全部 45 个候选的参考轨迹哈希无精确重复；CUDA 上将位姿量化至 1e−5 精度后仍为 32 条不同参考，官方 MotionLoader/FK 检查通过。后续由主实验准备脚本按整条轨迹划分为 24 条训练、4 条验证、4 条测试，并构建统一评估集。所有参考均来自实际反馈控制＋原生电机 PD 执行状态，没有使用旧 PPO 策略冒充 Agent 生成。

入库片段的有效前进速度范围为 **{lo:.4f}–{hi:.4f} m/s**。这些数值是数据采集控制器的结果，不是后续 SuperMimic 策略测试结果，也不是独立样本的总体成功率。删除足端跟踪项和关节限位惩罚后的训练效果由主实验单独报告。

| 阶段 | 程序版本数 | 通过数 | API 尝试调用数 | 用时（秒） |
|---|---:|---:|---:|---:|
'''
    for h in history:
        text += f'| {Path(h["run"]).name} | {h["attempts"]} | {h["accepted_count"]} | {h["attempted_api_calls"]} | {h["elapsed_s"]:.1f} |\n'
    text += f'''
上表含试采与中断后修订提示的探索成本，合计 **{summary['data_preparation_elapsed_s']/60:.2f} 分钟**；只有最后一阶段的 32 条用于本次数据集。中断阶段因重复出现关节相位分配和摆动相位归一化错误而停止，失败记录保留，未放宽检验门槛。API 完成回复共 {summary['completed_api_records']} 次，记录输入 {summary['prompt_tokens']} tokens、输出 {summary['completion_tokens']} tokens；中断且未返回的调用不具备 token 计数。

原始记录及32条导出数据：`runs/deepseek_report_seeded_20261115/`；主实验划分以其数据协议为准；逐次检查指标：`reports/course_draft_20261115/deepseek/accepted_trial_metrics.csv`；成本和去重审计：`reports/course_draft_20261115/deepseek/collection_audit.json`。
'''
    (OUT / 'section_deepseek_data.md').write_text(text)
    print(json.dumps({k:summary[k] for k in ['accepted_unique_references','accepted_physics_trials','data_preparation_elapsed_s','attempted_api_calls','cross_source_exact_reference_duplicates']},ensure_ascii=False))


if __name__ == '__main__':
    main()
