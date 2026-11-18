#!/usr/bin/env python3
"""Redraw the current course report from archived scalar data, without training.

Run: python3 scripts/plot_course_report_paper_style.py
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import zipfile
import shlex
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.ticker import NullLocator
from course_paper_plot_utils import (
    ROOT, SOURCE, OUT, HALF_LIFE_STEPS, BLUE, ORANGE, GRAY, COLORS,
    plt, np, configure, panel, draw_curve, curve_records, set_report_source,
    add_detail_view,
)

LABELS = {'ppo': 'PPO', 'ppo_extended': 'PPO (extended budget)',
          'supermimic': 'SuperMimic (official code)',
          'supermimic_deepseek': 'SuperMimic (DeepSeek)',
          'supermimic_codex': 'SuperMimic (Codex)'}
ORDER = ['ppo', 'supermimic', 'supermimic_deepseek', 'supermimic_codex']
FIELDS = [('return', 'Validation return', 'Return'),
          ('velocity_rmse_m_s', 'Velocity tracking', 'RMSE (m/s)'),
          ('heading_rms_deg', 'Heading tracking', 'RMS error (deg)'),
          ('root_position_rmse_m', 'Root position tracking', 'RMSE (m)')]
STYLE = {'supermimic_deepseek': '-', 'supermimic_codex': '-'}
TIME_LABEL = 'Training time after startup (min)'


def read_csv(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def startup_record(directory, method):
    """Use explicit new-run timing while retaining legacy update-log support."""
    update_path = directory/'updates.jsonl'
    first = json.loads(update_path.read_text().splitlines()[0])
    startup_s = float(first['training_s'])
    timing_source = update_path
    dependencies = [update_path]
    for name in ('startup.json', 'status.json'):
        path = directory/name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        value = payload.get('startup_training_s', payload.get('startup_s'))
        if value is None and name == 'startup.json':
            value = payload.get('training_s')
        if value is None:
            continue
        recorded = float(value)
        if not np.isclose(recorded, startup_s, rtol=1e-9, atol=1e-7):
            raise ValueError(f'{method}: explicit startup timing differs from first update')
        startup_s, timing_source = recorded, path
        dependencies.append(path)
        break
    protocol_path = directory/'protocol.json'
    run_protocol = json.loads(protocol_path.read_text())
    dependencies.append(protocol_path)
    interval = int(run_protocol['args']['eval_every'])
    return dict(method=method, seed=int(run_protocol['args']['seed']),
                startup_s=startup_s, first_logged_update_real_steps=int(first['real_steps']),
                source=str(timing_source.relative_to(ROOT)),
                update_source=str(update_path.relative_to(ROOT)),
                eval_every_real_steps=interval,
                first_update_evaluation=bool(run_protocol['args'].get('eval_after_first_update', False)),
                rule='First valid parameter-update log timestamp; retain observations at or after it.'), dependencies


def elapsed_after_startup(rows, startup_s):
    """Retain measured time coordinates; check optional explicit dense-run timing."""
    seconds = np.asarray([float(row['training_s']) for row in rows])
    elapsed = seconds-startup_s
    for row, value in zip(rows, elapsed):
        explicit = row.get('post_startup_training_s')
        if explicit not in (None, '', 'None', 'null'):
            if not np.isclose(float(explicit), value, rtol=1e-9, atol=1e-7):
                raise ValueError('Recorded post-startup timing does not match total minus startup')
    return seconds, elapsed


def report_figure_entries(text):
    """Associate each caption with its preceding Markdown image reference."""
    figures, numbers = {}, set()
    pending = None
    for paragraph in text.split('\n\n'):
        block = paragraph.strip()
        image = re.fullmatch(r'!\[[^\]]*\]\(([^)]+)\)', block)
        if image:
            pending = image.group(1)
            continue
        caption = re.fullmatch(r'图 (\d+)：(.*)', block, flags=re.DOTALL)
        if not caption:
            continue
        if pending is None:
            raise ValueError('Figure caption has no preceding image in report.md')
        number, body = caption.groups()
        source = pending.removeprefix('paper_style/')
        stem = str(Path(source).with_suffix(''))
        if number in numbers or stem in figures:
            raise ValueError('Duplicate figure number or image in report.md')
        numbers.add(number)
        figures[stem] = dict(number=number, caption=body, source=pending)
        pending = None
    return figures


def main():
    global SOURCE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE,
                        help='Report directory containing report.md and archived scalar CSVs')
    parser.add_argument('--suite', type=Path, default=ROOT/'runs/course_comparison_20261115',
                        help='Seven-run experiment directory used by the world-model figures')
    parser.add_argument('--extended-ppo', type=Path, default=ROOT/'runs/course_ppo_extended_20261115',
                        help='Independent longer-budget PPO run directory')
    args = parser.parse_args()
    SOURCE, OUT = set_report_source(args.source)
    suite, extended_directory = args.suite.resolve(), args.extended_ppo.resolve()
    configure(); OUT.mkdir(parents=True, exist_ok=True)
    report_text = (SOURCE/'report.md').read_text()
    report_figures = report_figure_entries(report_text)
    report_manifest = SOURCE/'report_manifest.json'
    report_settings = json.loads(report_manifest.read_text()) if report_manifest.is_file() else {}
    omit_high_speed = bool(report_settings.get('omit_high_speed', False))
    baseline_training_bands = bool(report_settings.get('baseline_training_uncertainty_bands', True))
    baseline_training_time = bool(report_settings.get('baseline_training_time_comparison', True))
    excluded_figures = {'improvements/speed_comparison', 'improvements/motion_training_curves'} if omit_high_speed else set()
    if excluded_figures.intersection(report_figures):
        raise ValueError('High-speed figures remain in report.md despite omit_high_speed=true')
    supplemental_figures = {'overview_paper_layout', 'improvements/motion_training_curves'} - excluded_figures
    sources = [SOURCE / f for f in ['supermimic_curve_data.csv',
                'supermimic_test_results.csv', 'ppo_extended_curve_data.csv',
                'ppo_extended_time_points.csv', 'report.md']]
    before = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    curves = read_csv(SOURCE / 'supermimic_curve_data.csv')
    final = {r['method']: r for r in read_csv(SOURCE / 'supermimic_test_results.csv')}
    extended = read_csv(SOURCE / 'ppo_extended_curve_data.csv')
    records, saved, captions, gap_audit, style_audit = [], [], {}, [], []
    trajectory_assets = []
    startup, startup_audit = {}, []
    for method in sorted({r['method'] for r in curves + extended}):
        directory = extended_directory if method == 'ppo_extended' else suite/method
        entry, dependencies = startup_record(directory, method)
        startup[method] = entry['startup_s']
        assert startup[method] > 0
        sources.extend(dependencies)
        method_rows = [r for r in curves + extended if r['method'] == method]
        _, method_elapsed = elapsed_after_startup(method_rows, startup[method])
        entry['zero_time_evaluation_observed'] = bool(np.any(np.isclose(method_elapsed, 0., atol=1e-7)))
        if entry['first_update_evaluation'] and not entry['zero_time_evaluation_observed']:
            raise ValueError(f'{method}: requested first-update evaluation is missing')
        startup_audit.append(entry)
    eval_intervals = {row['method']:row['eval_every_real_steps'] for row in startup_audit}
    first_update_evaluations = {row['method']:row['first_update_evaluation'] for row in startup_audit}
    with PdfPages(OUT / 'all_figures.pdf') as book:
        def save(fig, name):
            if name in excluded_figures or (name not in report_figures and name not in supplemental_figures):
                plt.close(fig)
                return
            path = OUT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            def all_axes(axes):
                for ax in axes:
                    yield ax
                    yield from all_axes(ax.child_axes)
            audited_axes = list(all_axes(fig.axes))
            lines = [line for ax in audited_axes for line in ax.lines]
            assert all(line.get_linestyle() == '-' for line in lines), name
            assert all(line.get_marker() in (None, 'None', '', ' ') for line in lines), name
            assert not any(isinstance(item, PathCollection)
                           for ax in audited_axes for item in ax.collections), name
            bands = sum(isinstance(item, PolyCollection)
                        for ax in audited_axes for item in ax.collections)
            if name == 'baselines/baseline_learning_curves' and not baseline_training_bands:
                assert bands == 0 and len(lines) == (18 if baseline_training_time else 9), name
            style_audit.append(dict(figure=name, solid_lines=len(lines),
                                    dashed_lines=0, marker_lines=0, scatter_collections=0,
                                    uncertainty_bands=bands,
                                    detail_views=[ax._curve_detail for ax in audited_axes
                                                  if hasattr(ax, '_curve_detail')]))
            # PNG is used in the Chinese report for portable font embedding;
            # PDF and SVG are separately exported for publication/editing.
            for suffix in ('png', 'pdf', 'svg'):
                fig.savefig(path.with_suffix('.' + suffix), bbox_inches='tight',
                            pad_inches=.09, facecolor='white')
            book.savefig(fig, bbox_inches='tight', pad_inches=.09)
            saved.append(name)
            plt.close(fig)

        import course_paper_baselines
        # Also refresh the binding if main() is called twice in one process.
        course_paper_baselines.SOURCE = SOURCE
        base = course_paper_baselines.draw_all(save, omit_high_speed=omit_high_speed,
                                              training_uncertainty=baseline_training_bands,
                                              training_time_comparison=baseline_training_time)
        captions.update(base['captions'])
        sources.extend(base['sources'])
        records.extend(base['curve_records'])
        startup_audit.extend(base['startup_audit'])

        if 'reference_execution_comparison' in report_figures:
            import course_paper_reference_trajectory
            trajectory = course_paper_reference_trajectory.draw(save, OUT)
            sources.extend(trajectory['sources'])
            trajectory_assets.extend(trajectory['assets'])

        # Subtract each run's first valid update timestamp; never synthesize a zero-time score.
        for ablation in (False, True):
            methods = (['supermimic', 'supermimic_h16', 'supermimic_h64',
                        'supermimic_no_critic'] if ablation else ORDER)
            labels = ({'supermimic': r'$H=32$, critic',
                       'supermimic_h16': r'$H=16$, critic',
                       'supermimic_h64': r'$H=64$, critic',
                       'supermimic_no_critic': r'$H=32$, no critic'}
                      if ablation else LABELS)
            colors = ({'supermimic': BLUE, 'supermimic_h16': '#00866b',
                       'supermimic_h64': '#a3468c', 'supermimic_no_critic': ORANGE}
                      if ablation else COLORS)
            name = 'supermimic_ablation' if ablation else 'supermimic_comparison'
            fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
            fig.subplots_adjust(left=.085, right=.985, bottom=.13, top=.85,
                                hspace=.65, wspace=.30)
            for i, (ax, (metric, title, ylabel)) in enumerate(zip(axes.flat, FIELDS)):
                panel(ax, chr(97+i), title, TIME_LABEL, ylabel)
                for method in methods:
                    rows = [r for r in curves if r['method'] == method]
                    steps = [int(r['real_steps']) for r in rows]
                    seconds, elapsed = elapsed_after_startup(rows, startup[method])
                    raw = [float(r[metric]) for r in rows]
                    sm, gaps = draw_curve(ax, steps, elapsed/60, raw,
                        colors[method], labels[method], gap_detection=True, start_x=0.)
                    records.extend(curve_records(name, method, 42, metric,
                                                 steps, seconds, raw, sm, startup_s=startup[method]))
                    if i == 0:
                        gap_audit.append(dict(figure=name, method=method,
                            gap_endpoint_indices=gaps, observed_points=len(rows),
                            displayed_points=int(np.sum(seconds >= startup[method]))))
                if metric == 'return':
                    ax.set_ylim(0, 30)
                    if ablation:
                        add_detail_view(ax, bounds=[.41, .12, .56, .36],
                                        xlim=(1.8, 2.8), ylim=(20.0, 22.1))
                    else:
                        add_detail_view(ax, bounds=[.41, .12, .56, .25],
                                        xlim=(2.4, 3.1), ylim=(22.0, 22.5),
                                        labels=[LABELS['supermimic_deepseek'], LABELS['supermimic_codex']],
                                        title='Agent detail (min)')
                else:
                    shown = np.concatenate([line.get_ydata() for line in ax.lines])
                    lo, hi = float(shown.min()), float(shown.max())
                    margin = max((hi-lo)*.12, hi*.015)
                    ax.set_ylim(max(0., lo-margin), hi+margin)
                ax.set_xlim(left=0)
            handles, legend_labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc='upper center', ncol=2,
                       bbox_to_anchor=(.535, .995), fontsize=11, handlelength=2.7)
            fig.text(.085, .018,
                'Startup excluded; origin: first logged update. EMA half-life: 131,072 real steps. Single training seed.',
                fontsize=10, color='#555555')
            save(fig, name)

        # Extended PPO uses the same post-startup display rule; total-time tables remain separate.
        fig, (ax, detail_ax) = plt.subplots(1, 2, figsize=(11.2, 4.5),
                                           gridspec_kw={'width_ratios': [1.65, 1]})
        fig.subplots_adjust(left=.08, right=.985, bottom=.24, top=.76, wspace=.30)
        panel(ax, 'a', 'Full training', TIME_LABEL, 'Validation return')
        panel(detail_ax, 'b', 'Late-stage detail', 'Training time (min)', 'Validation return')
        for method in ['ppo_extended', *ORDER[1:]]:
            rows = [r for r in extended if r['method'] == method]
            steps = [int(r['real_steps']) for r in rows]
            seconds, elapsed = elapsed_after_startup(rows, startup[method])
            raw = [float(r['validation_return']) for r in rows]
            sm, gaps = draw_curve(ax, steps, elapsed/60, raw, COLORS[method], LABELS[method],
                                  gap_detection=True, start_x=0.)
            keep = elapsed >= 0
            detail_ax.plot(elapsed[keep]/60, sm[keep], color=COLORS[method],
                           label=LABELS[method], lw=1.8, ls='-', marker=None)
            records.extend(curve_records('ppo_extended_time_comparison', method, 42,
                            'return', steps, seconds, raw, sm, startup_s=startup[method]))
            gap_audit.append(dict(figure='ppo_extended_time_comparison', method=method,
                            gap_endpoint_indices=gaps, observed_points=len(rows),
                            displayed_points=int(np.sum(seconds >= startup[method]))))
        ax.set_ylim(0, 30); ax.set_xlim(left=0)
        detail_ax.set(xlim=(1.7, 3.15), ylim=(20.0, 22.7))
        detail_ax._curve_detail = dict(kind='zoom', xlim=[1.7, 3.15], ylim=[20., 22.7],
                                      coordinates='Exact copies of displayed EMA line coordinates')
        handles, legend_labels = ax.get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper center', ncol=2,
                   bbox_to_anchor=(.53, 1.), fontsize=11, handlelength=2.8)
        fig.text(.08, .035,
            'Startup excluded for every method; origin: first logged update. EMA half-life: 131,072 real steps.',
            fontsize=10, color='#555555')
        save(fig, 'ppo_extended_time_comparison')

        # Optional overview uses the supplied paper's large-left/two-right layout.
        fig = plt.figure(figsize=(12, 4.8))
        gs = fig.add_gridspec(2, 2, left=.075, right=.985, top=.88, bottom=.16,
                             width_ratios=[1.25, 1], hspace=.85, wspace=.32)
        ax = fig.add_subplot(gs[:, 0])
        panel(ax, 'a', 'Validation return', TIME_LABEL, 'Return')
        for method in ORDER:
            rows = [r for r in curves if r['method'] == method]
            steps = [int(r['real_steps']) for r in rows]
            seconds, elapsed = elapsed_after_startup(rows, startup[method])
            raw = [float(r['return']) for r in rows]
            draw_curve(ax, steps, elapsed/60, raw, COLORS[method], LABELS[method],
                       gap_detection=True, start_x=0.)
        ax.set_ylim(0, 30); ax.set_xlim(left=0)
        ax.legend(loc='lower left', fontsize=10.5, handlelength=2.8)
        for row, field, title, ylabel in [
                (0, 'velocity_rmse_m_s', 'Final velocity error', 'RMSE (m/s)'),
                (1, 'heading_rms_deg', 'Final heading error', 'RMS (deg)')]:
            right = fig.add_subplot(gs[row, 1])
            panel(right, chr(98+row), title, '', ylabel)
            right.plot(range(4), [float(final[method][field]) for method in ORDER],
                       color=BLUE, lw=2.3, ls='-', marker=None,
                       solid_capstyle='round', solid_joinstyle='round', zorder=3)
            right.set_xlim(-.35, 3.35)
            right.set_ylim(0, max(float(final[m][field]) for m in ORDER)*1.20)
            right.set_xticks(range(4), ['PPO', 'Official', 'DeepSeek', 'Codex'])
            right.xaxis.set_minor_locator(NullLocator())
        fig.text(.075, .025,
            'Left: post-startup EMA curves. Right: final test metrics. One training seed per method.',
            fontsize=10, color='#555555')
        save(fig, 'overview_paper_layout')

    intervals = sorted(set(eval_intervals.values()))
    interval_note = (f'周期评估间隔为 {intervals[0]:,} 个真实交互步。' if len(intervals) == 1 else
                     '各组周期评估间隔分别记录于附带的绘图协议。')
    origin_note = ('各组均在首次训练更新后立即进行实测评估，作为零时刻。'
                   if all(first_update_evaluations.values()) else
                   '曲线从保留范围内的首条实测评估开始。')
    # The report owns numbering and captions; deleted sections can renumber
    # images without assigning a world-model caption to an unrelated figure.
    main_figs = [name for name in saved if name in report_figures]
    captions = {entry['number']:entry['caption'] for entry in report_figures.values()}
    protocol = dict(
        version=5, style_reference='User-provided SuperMimic paper figure; continuous post-startup curves',
        report_source=str(SOURCE.relative_to(ROOT)),
        omit_high_speed=omit_high_speed,
        report_figure_numbers={name:entry['number'] for name, entry in report_figures.items()},
        run_suite=str(suite.relative_to(ROOT)), extended_ppo=str(extended_directory.relative_to(ROOT)),
        evaluation_interval_real_steps=eval_intervals,
        first_update_evaluation=first_update_evaluations,
        training_curve_smoothing=dict(method='causal EMA', half_life_real_steps=HALF_LIFE_STEPS,
            alpha='1 - 2 ** (-(steps[i] - steps[i-1]) / half_life_real_steps)',
            initialization='first recorded value',
            artificial_endpoint_forcing=False, interpolation=False, extrapolation=False),
        uncertainty=(('Baseline training curves: mean +/- sample SD (ddof=1). ' if baseline_training_bands else
                      'Baseline training curves: one mean line per method; no uncertainty bands. ')+
                     'Means use three separately smoothed training seeds at shared retained steps'+
                     ('; time coordinates are the mean of their recorded elapsed times minus individual startup offsets. ' if baseline_training_time else
                      '. Baseline training comparisons use environment steps only. ')+
                     'Baseline final-test figures retain sample SD bands. No confidence bands for single-seed experiments.'),
        baseline_training_uncertainty_bands=baseline_training_bands,
        baseline_training_time_comparison=baseline_training_time,
        time_origin='Each run uses its first valid parameter-update log timestamp. Drop earlier observations, then display recorded training_s minus startup_s. Evaluation time is excluded by the training logger. Dense runs include a measured first-update evaluation at time zero; older runs start at the first retained measurement. No synthetic zero-time observation.',
        crop_order='Compute EMA on the full recorded history, then crop; no reinitialization of EMA.',
        startup_boundaries=startup_audit,
        raw_measurements='Preserved in display_curve_data.csv; not overlaid on the plots.',
        curve_rendering='One continuous solid EMA line per method, no markers, scatter, dashed connectors or raw auxiliary lines. Adjacent recorded EMA values are joined without resampling.',
        readability=dict(palette='Distinct blue, orange, green and purple for four-method comparisons',
                         zooms='Copy existing EMA line coordinates; bounds explicitly labeled and audited',
                         error_axes='Bounds follow the full observed EMA range plus padding',
                         smoothness='Absolute means plus 100 * (smooth - original) / original insets',
                         vertical_offsets=False, altered_smoothing=False),
        gaps='Intervals longer than three times the median interval remain in gap_audit. All intervals use solid lines between recorded EMA values; no additional evaluations are introduced.',
        test_results='No smoothing or changes to final test values, sample SD or report tables.',
        units='Keep Return and minutes. Plot time excludes startup; all cost and matched-total-time tables retain original inclusive timing.',
        baseline_heading='Archived baseline CSV values already in degrees; not multiplied again.',
        colors=COLORS, seed_counts={'historical_baselines':3, 'current_supermimic_and_improvements':1},
        no_training_or_model_computation=True)
    (OUT/'smoothing_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    write_csv(OUT/'display_curve_data.csv', records)
    # Source records remain the authority; plotting is a reversible presentation transform.
    sources = sorted(set(sources))
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    assert all(hashes[k] == v for k, v in before.items())
    for r in records:
        assert np.isfinite(r['raw_value']) and np.isfinite(r['smoothed_value'])
        if 'rmse' in r['metric'] or 'rms' in r['metric']:
            assert r['raw_value'] >= 0 and r['smoothed_value'] >= 0
    manifest = dict(report_source=str(SOURCE.relative_to(ROOT)),
        omit_high_speed=omit_high_speed,
        baseline_training_uncertainty_bands=baseline_training_bands,
        baseline_training_time_comparison=baseline_training_time,
        report_figure_numbers={name:entry['number'] for name, entry in report_figures.items()},
        run_suite=str(suite.relative_to(ROOT)), extended_ppo=str(extended_directory.relative_to(ROOT)),
        source_sha256=hashes, script_sha256={
        str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        'scripts/course_paper_plot_utils.py':sha(ROOT/'scripts/course_paper_plot_utils.py'),
        'scripts/course_paper_baselines.py':sha(ROOT/'scripts/course_paper_baselines.py')},
        figures=saved, main_figures=main_figs, display_record_count=len(records),
        output_sha256={str(p.relative_to(OUT)):sha(p) for name in saved
                       for p in [OUT/(name+'.png'),OUT/(name+'.pdf'),OUT/(name+'.svg')]},
        gap_audit=gap_audit, style_audit=style_audit, startup_audit=startup_audit, raw_source_data_unchanged=True,
        source_measurement_x_positions_preserved_in_csv=True,
        baseline_time_coordinates=('Mean post-startup training time across three seeds at shared retained steps.' if baseline_training_time else
                                   'Not plotted; recorded times and startup offsets remain in display_curve_data.csv.'),
        protocol_file='smoothing_protocol.json', caption_source='report.md')
    if trajectory_assets:
        manifest['script_sha256']['scripts/course_paper_reference_trajectory.py'] = sha(
            ROOT/'scripts/course_paper_reference_trajectory.py')
        manifest['output_sha256'].update({name:sha(OUT/name) for name in trajectory_assets})
        manifest['reference_execution'] = trajectory['metadata']
    (OUT/'plot_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    redraw_command = shlex.join(['python3', 'scripts/plot_course_report_paper_style.py',
        '--source', str(SOURCE.relative_to(ROOT)), '--suite', str(suite.relative_to(ROOT)),
        '--extended-ppo', str(extended_directory.relative_to(ROOT))])
    plot_notes = '''# 绘图说明

仿照用户提供论文图的衬线字体、蓝橙配色、细坐标轴、无背景网格、左上角分面标题；额外提供大左图与右侧上下两图的 overview_paper_layout。图中的英文使用直观指标名称，中文解释保留在报告图注。

训练曲线统一按真实交互步数作因果 EMA：半衰期 131072 步，alpha = 1 - 2^(-步数差 / 131072)。每种方法以一条连续实线连接平滑后的记录值，不显示点标记、虚线和原始辅助线。该平滑参数对所有算法一致，不因记录密度而改变训练跨度。原始值保留于附带 CSV，不新增采样、不外推、不调整 EMA 端点。

每组以首次记录到有效参数更新的日志时间为截取边界，仅显示边界及之后的观测，时间横坐标为原累计训练时间减去该边界时间。该边界受日志频率限制，不等同于精确首次优化器步骤结束时刻。所有方法采用同一规则，不能将扣除的全部时段解释为纯编译时间。评估用时不计入横轴；EMA先用完整历史计算，再裁剪显示，不补造零时刻回报。

__EVALUATION_NOTE__

基线先分别平滑三个训练种子，再在共同保留步数上计算均值 ± 样本标准差（ddof=1）。步数图保留绝对真实交互步数，时间图采用该步数处三个种子各自扣除启动时段后的时间均值，两者均以实线与标准差阴影展示。单种子实验不绘制跨种子误差带。原日志回报窗口为 PPO 最近100回合、DQN/SAC最近256回合。

所有保留时间间隔均以实线连接已有记录；连线用于呈现趋势，间隔内不增加评估数据。Return 保持原始量纲，时间单位为分钟。最终测试图按评估类别连接原始均值，基线测试保留样本标准差阴影；类别间连线仅辅助比较。最终测试数据、表格和正文比较数值均不平滑；计算成本及相近总时间对照表仍使用含启动开销的原始计时。补充 PPO 三组训练曲线使用各自训练奖励，组间比较采用共同测试奖励。

重画：在 Sesame 根目录运行 `__REDRAW_COMMAND__`。`display_curve_data.csv` 保留所有原始值、EMA值、实际步数、原始时间和seed，并增加startup_s、plot_training_s及included列。`smoothing_protocol.json` 保存各组边界与规则，`plot_manifest.json` 保存输入和图片哈希。PDF/SVG为独立矢量图，PNG为320dpi报告插图。绘图阶段仅处理已完成的训练记录。
'''
    if omit_high_speed:
        plot_notes = plot_notes.replace('补充 PPO 三组训练曲线使用各自训练奖励，组间比较采用共同测试奖励。', '')
    if not baseline_training_bands:
        plot_notes = plot_notes.replace('再在共同保留步数上计算均值 ± 样本标准差（ddof=1）',
                                        '再在共同保留步数上计算均值')
        plot_notes = plot_notes.replace('两者均以实线与标准差阴影展示',
                                        '两者均按每种方法一条均值实线展示，不绘制误差带')
    if not baseline_training_time:
        plot_notes = plot_notes.replace(
            '步数图保留绝对真实交互步数，时间图采用该步数处三个种子各自扣除启动时段后的时间均值，两者均',
            '基线训练曲线仅按绝对真实交互步数展示，不作训练时间横轴对比，曲线均')
    if trajectory_assets:
        plot_notes += ('\n参考与控制器执行轨迹采用同一段 6 秒、50 Hz 原始记录，连续实线直接连接各帧，'
                       '不作 EMA 平滑、时间重对齐或端点修正。CSV 保存全部 301 帧，JSON 保存来源与校验值。'
                       '这是参考采集阶段的控制器仿真检验，不是训练后策略推理。\n')
    (OUT/'plot_protocol.md').write_text(plot_notes.replace('__EVALUATION_NOTE__', interval_note+origin_note)
                                       .replace('__REDRAW_COMMAND__', redraw_command))
    overrides = dict(figure_sources={s+'.png':'paper_style/'+s+'.png' for s in main_figs},
        caption_overrides=captions,
        link_sources=({'improvements/motion_training_curves.pdf':
                       'paper_style/improvements/motion_training_curves.pdf'}
                      if 'improvements/motion_training_curves' in saved else {}),
        extra_assets=['paper_style/'+s for s in ('plot_protocol.md', 'smoothing_protocol.json',
                       'display_curve_data.csv', 'plot_manifest.json')],
        note=('Continuous post-startup curves without markers or raw overlays. Every run is cropped at its first valid update log and time is rebased to that timestamp. EMA half-life remains 131072 real steps, calculated before cropping. '+
              ('Baseline training curves show three-seed mean and sample SD. ' if baseline_training_bands else
               'Baseline training curves show only the three-seed mean; final-test figures retain sample SD. ')+
              ('' if baseline_training_time else 'Baseline training comparisons use environment steps only. ')+
              'Raw data and all test/table values remain unchanged; cost tables include startup.'))
    overrides['extra_assets'].extend('paper_style/'+name for name in trajectory_assets)
    (OUT/'report_overrides.json').write_text(json.dumps(overrides,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'paper_style_figures.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in saved:
            for suffix in ('.png','.pdf','.svg'):
                archive.write(OUT/(name+suffix), name+suffix)
        for name in ('all_figures.pdf','display_curve_data.csv','smoothing_protocol.json',
                     'plot_manifest.json','plot_protocol.md','report_overrides.json'):
            archive.write(OUT/name,name)
        for name in trajectory_assets:
            archive.write(OUT/name, name)
    print(json.dumps(dict(output=str(OUT),figures=len(saved),display_records=len(records)),ensure_ascii=False))


if __name__ == '__main__':
    main()
