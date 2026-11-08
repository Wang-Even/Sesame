#!/usr/bin/env python3
"""Plot saved Sesame domain-randomization JSON metadata; never import a model or simulator."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / '.runtime/matplotlib'
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(MPL_CACHE)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

POLICIES = ('zero_residual', 'baseline_ppo', 'dr_ppo')
LABELS = {'zero_residual': 'Zero residual', 'baseline_ppo': 'Fixed-domain PPO', 'dr_ppo': 'DR PPO'}
COLORS = {'zero_residual': '#94A3B8', 'baseline_ppo': '#2563EB', 'dr_ppo': '#D97706'}
CONDITIONS = ('nominal', 'randomized')
CONDITION_LABELS = {'nominal': 'Nominal physics', 'randomized': 'Randomized physics'}


def read_json(path, inputs):
    raw = path.read_bytes()
    inputs[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return json.loads(raw)


def read_rows(path, inputs):
    if not path.exists():
        return []
    raw = path.read_bytes()
    inputs[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    lines = raw.decode().splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A growing log may contain one unfinished final line.
            if index != len(lines)-1:
                raise
    return rows


def smooth(values, window):
    return [statistics.mean(values[max(0, index-window+1):index+1]) for index in range(len(values))]


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def save_figure(fig, output, name):
    paths = []
    for suffix in ('png', 'pdf'):
        path = output / f'{name}.{suffix}'
        fig.savefig(path, dpi=200, facecolor='white')
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_learning(training, output, window, status):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    specifications = [
        ('Training episode return', 'mean_episode_return', None, False),
        ('Training episode displacement (m)', None, 'Episode/distance_x_m', False),
        ('Training falls (%)', None, 'Episode/fall_fraction', True),
        ('PPO value-function loss', 'value_loss', None, False),
    ]
    for axis, (title, row_key, episode_key, percent) in zip(axes.flat, specifications):
        plotted = False
        for label in ('baseline_ppo', 'dr_ppo'):
            rows = training.get(label, [])
            points = []
            for row in rows:
                value = (row.get('losses', {}).get('value_function') if row_key == 'value_loss'
                         else row.get(row_key) if row_key else row.get('episode_metrics', {}).get(episode_key))
                if finite_number(value) and finite_number(row.get('environment_transitions')):
                    points.append((row['environment_transitions']/1e6, value*100 if percent else value, row['iteration']))
            if not points:
                continue
            xs, ys, iterations = zip(*points)
            smoothed = [statistics.mean(y for y, it in zip(ys, iterations) if iteration-window < it <= iteration)
                        for iteration in iterations]
            axis.plot(xs, ys, color=COLORS[label], alpha=0.2, linewidth=0.8)
            axis.plot(xs, smoothed, color=COLORS[label], linewidth=1.9,
                      marker='o' if len(xs) < 3 else None, label=LABELS[label])
            plotted = True
        axis.set(title=title, xlabel='Environment transitions (millions)')
        axis.grid(alpha=0.18)
        if percent:
            axis.set_ylim(-3, 103)
        if row_key == 'value_loss' and plotted:
            axis.set_yscale('log')
        if not plotted:
            axis.text(0.5, 0.5, 'No completed episodes yet', ha='center', va='center', transform=axis.transAxes)
    handles = [Patch(facecolor=COLORS[label], label=LABELS[label]) for label in ('baseline_ppo', 'dr_ppo')
               if training.get(label)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    dr_updates = len(training.get('dr_ppo', []))
    fig.suptitle(f'Sesame PPO learning curves | DR: {dr_updates} update{"s" if dr_updates != 1 else ""} | {status}', fontsize=15, y=0.99)
    fig.text(0.5, 0.017,
             f'Raw values are faint; solid lines show a trailing {window}-update mean. '
             'Training domains differ; use the paired evaluation for comparison.',
             ha='center', fontsize=9, color='#475569')
    fig.subplots_adjust(top=0.84, bottom=0.12, hspace=0.42, wspace=0.24)
    return save_figure(fig, output, 'dr_learning_curves')


def plot_evaluation(evaluation, output, dr_updates):
    available = [label for label in POLICIES if any(label in evaluation.get(c, {}) for c in CONDITIONS)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    specifications = [
        ('Forward displacement per episode (m)', 'distance_x_m', True, False),
        ('Observed falls (%)', 'fall_rate', False, True),
        ('Absolute final heading error (degrees)', 'abs_final_heading_yaw_deg', True, False),
        ('Local evaluation return', 'return', True, False),
    ]
    width = 0.22
    for axis, (title, key, errorbar, percent) in zip(axes.flat, specifications):
        for policy_index, label in enumerate(available):
            xs, means, spreads = [], [], []
            for condition_index, condition in enumerate(CONDITIONS):
                document = evaluation.get(condition, {}).get(label)
                if not document:
                    continue
                value = document['metrics'].get(key)
                if errorbar:
                    if not isinstance(value, dict) or not finite_number(value.get('mean')):
                        continue
                    mean, spread = value['mean'], value.get('std', 0.0)
                else:
                    if not finite_number(value):
                        continue
                    mean, spread = value, 0.0
                xs.append(condition_index+(policy_index-(len(available)-1)/2)*width)
                means.append(mean*100 if percent else mean)
                spreads.append(spread*100 if percent else spread)
            if not xs:
                continue
            axis.bar(xs, means, width=width*0.88, color=COLORS[label],
                     yerr=spreads if errorbar else None, capsize=3,
                     error_kw={'elinewidth': 1.1, 'capthick': 1.1}, label=LABELS[label])
            for x, value, spread in zip(xs, means, spreads):
                text = f'{value:.0f}%' if percent else f'{value:.2f}' if key=='distance_x_m' else f'{value:.1f}'
                axis.annotate(text, (x, value+spread if value>=0 else value-spread),
                              textcoords='offset points', xytext=(0, 5 if value>=0 else -12),
                              ha='center', fontsize=8)
        axis.set_title(title, pad=12)
        axis.set_xticks(range(len(CONDITIONS)), [CONDITION_LABELS[c] for c in CONDITIONS])
        axis.grid(axis='y', alpha=0.18)
        axis.set_axisbelow(True)
        axis.axhline(0, color='#64748B', linewidth=0.65)
        if percent:
            axis.set_ylim(0, 105)
        elif key=='abs_final_heading_yaw_deg':
            axis.set_ylim(0, 190)
            axis.axhline(45, linestyle=':', color='#64748B', linewidth=1)
        else:
            axis.margins(y=0.18)
    counts = sorted({int(doc['metrics']['episodes']) for values in evaluation.values() for doc in values.values()})
    seeds = sorted({int(doc['metrics']['seed']) for values in evaluation.values() for doc in values.values()})
    fig.suptitle(f'Sesame PPO | paired held-out evaluation | DR: {dr_updates} update{"s" if dr_updates != 1 else ""}', fontsize=15, y=0.99)
    fig.legend(handles=[Patch(facecolor=COLORS[label], label=LABELS[label]) for label in available],
               loc='upper center', bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=False)
    fig.text(0.5, 0.025,
             f'{"/".join(map(str, counts))} episodes per policy/domain; seed {"/".join(map(str, seeds))}. '
             'Bars: episode mean; error bars: 1 SD, not confidence intervals.\n'
             'One training seed per learned policy. Heading is relative to world +x; reward was not changed to fix sideways motion.',
             ha='center', fontsize=9, color='#475569')
    fig.subplots_adjust(top=0.84, bottom=0.14, hspace=0.43, wspace=0.26)
    return save_figure(fig, output, 'dr_evaluation_comparison')


def paired_summary(evaluation):
    summary = {}
    for condition, documents in evaluation.items():
        first = next(iter(documents.values()))
        pairing = {label: document.get('initial_conditions') == first.get('initial_conditions')
                   for label, document in documents.items()}
        if not all(pairing.values()):
            raise ValueError(f'Initial states/domains differ across policies in {condition}; cannot label evaluation paired')
        values = {'matching_initial_states_and_domains': pairing,
                  'metrics': {label: document['metrics'] for label, document in documents.items()}}
        if 'dr_ppo' in documents and 'baseline_ppo' in documents:
            def indexed(document):
                return {(row['batch_seed'], row['env_id']): row for row in document['episodes']}
            dr = indexed(documents['dr_ppo'])
            baseline = indexed(documents['baseline_ppo'])
            if set(dr) != set(baseline):
                raise ValueError(f'Episode identifiers do not match for {condition}')
            deltas = [dr[key]['distance_x_m']-baseline[key]['distance_x_m'] for key in sorted(dr)]
            values['paired_dr_minus_baseline_distance_m'] = {
                'mean': statistics.mean(deltas), 'std_across_paired_episodes': statistics.pstdev(deltas),
                'pairs': len(deltas),
            }
        summary[condition] = values
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--baseline-run', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smooth-window', type=int, default=10)
    args = parser.parse_args()
    if args.smooth_window < 1:
        parser.error('smooth-window must be positive')
    run = args.run_dir.resolve()
    output = (args.output_dir or run / 'plots_dr').resolve()
    if not output.is_relative_to(ROOT):
        parser.error('output-dir must be inside the Sesame workspace')
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    result_path = run / 'result.json'
    result = read_json(result_path, inputs) if result_path.exists() else {}
    baseline = args.baseline_run
    if baseline is None and result.get('baseline_checkpoint'):
        baseline = Path(result['baseline_checkpoint']).parent
    training = {'dr_ppo': read_rows(run / 'training.jsonl', inputs)}
    if baseline is not None:
        training['baseline_ppo'] = read_rows(baseline.resolve() / 'training.jsonl', inputs)
    artifacts = plot_learning(training, output, args.smooth_window, result.get('status', 'unknown'))
    evaluation = {}
    for condition in CONDITIONS:
        for label in POLICIES:
            path = run / 'evaluation' / condition / f'{label}.json'
            if path.exists():
                evaluation.setdefault(condition, {})[label] = read_json(path, inputs)
    summary = paired_summary(evaluation) if evaluation else {}
    if evaluation:
        artifacts.extend(plot_evaluation(evaluation, output, len(training.get('dr_ppo', []))))
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'run_dir': str(run),
        'run_status_at_read': result.get('status', 'unknown'),
        'method': 'Saved JSON metadata only; no checkpoint loading, model execution, physics or Git.',
        'training_rows': {label: len(rows) for label, rows in training.items()},
        'evaluation_complete': all(set(evaluation.get(c, {})) == set(POLICIES) for c in CONDITIONS),
        'evaluation': summary, 'artifacts': artifacts, 'source_files': inputs,
        'limitations': ['Episode SD is not training-seed uncertainty or a confidence interval.',
                        'Training curves use different domains; compare paired held-out tests for policy effects.',
                        'The unchanged reward does not require a forward-facing body; heading diagnostics are shown.'],
    }
    report_path = output / 'plot_report.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report': str(report_path), 'artifacts': artifacts,
                      'evaluation_complete': report['evaluation_complete']}, indent=2))


if __name__ == '__main__':
    main()
