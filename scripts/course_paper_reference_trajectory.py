"""Plot recorded reference/controller trajectories; no model or simulation runs."""
import csv
import hashlib
import json

from course_paper_plot_utils import ROOT, BLUE, ORANGE, np, plt, panel


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_motion(path, execution=False):
    with np.load(path, allow_pickle=False) as data:
        pos = data['root_pos_w'].astype(float)
        quat = data['root_quat_w'].astype(float)
        fps = int(data['fps'])
    if execution:
        if pos.ndim != 3 or pos.shape[1] != 1:
            raise ValueError('Expected one recorded execution environment')
        pos, quat = pos[:, 0], quat[:, 0]
    if pos.shape != (301, 3) or quat.shape != (301, 4) or fps != 50:
        raise ValueError('Expected 301 aligned samples at 50 Hz')
    if not np.isfinite(pos).all() or not np.isfinite(quat).all():
        raise ValueError('Nonfinite trajectory data')
    if not np.allclose(np.linalg.norm(quat, axis=1), 1., atol=1e-5):
        raise ValueError('Invalid quaternion norms')
    w, x, y, z = quat.T
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return pos, yaw, np.arange(len(pos), dtype=float)/fps


def draw(save, out):
    run = ROOT/'runs/agent32_collection_20261114'
    video_file = run/'video.json'
    video = json.loads(video_file.read_text())
    attempt = ROOT/video['selected_attempt']
    ref_file, exe_file = attempt/'candidate_reference.npz', attempt/'nominal_execution.npz'
    report_file = attempt/'report.json'
    report = json.loads(report_file.read_text())
    if sha(ref_file) != video['reference_sha256'] or sha(exe_file) != video['execution_sha256']:
        raise ValueError('Recorded trajectories differ from the reviewed video')
    if not report['accepted'] or not video['accepted']:
        raise ValueError('Expected the accepted collection example')
    ref, ref_yaw, time = read_motion(ref_file)
    exe, exe_yaw, exe_time = read_motion(exe_file, execution=True)
    if not np.array_equal(time, exe_time) or not np.allclose(ref[0], exe[0], atol=1e-7):
        raise ValueError('Initial positions or timestamps differ')
    ref_disp, exe_disp = ref-ref[0], exe-exe[0]
    angle_error = np.arctan2(np.sin(exe_yaw-ref_yaw), np.cos(exe_yaw-ref_yaw))
    metrics = dict(
        reference_forward_displacement_m=float(ref_disp[-1, 0]),
        execution_forward_displacement_m=float(exe_disp[-1, 0]),
        reference_mean_forward_speed_m_s=float(ref_disp[-1, 0]/time[-1]),
        execution_mean_forward_speed_m_s=float(exe_disp[-1, 0]/time[-1]),
        execution_final_lateral_displacement_m=float(exe_disp[-1, 1]),
        execution_max_abs_lateral_displacement_m=float(np.max(np.abs(exe_disp[:, 1]))),
        root_position_rmse_m=float(np.sqrt(np.mean(np.sum((exe[1:]-ref[1:])**2, axis=1)))),
        heading_rms_deg=float(np.rad2deg(np.sqrt(np.mean(angle_error[1:]**2)))))
    # Independent metadata cross-checks; no filtering or endpoint adjustment.
    for key, measured in [('forward_displacement_m', metrics['execution_forward_displacement_m']),
                          ('world_body_yaw_rms_deg', metrics['heading_rms_deg'])]:
        if not np.isclose(measured, video['execution'][key], rtol=1e-5, atol=1e-6):
            raise ValueError(f'Trajectory/video metric mismatch: {key}')
    rows = [dict(time_s=float(t), reference_x_m=float(r[0]), execution_x_m=float(e[0]),
                 reference_y_m=float(r[1]), execution_y_m=float(e[1]),
                 reference_yaw_deg=float(np.rad2deg(ry)), execution_yaw_deg=float(np.rad2deg(ey)))
            for t, r, e, ry, ey in zip(time, ref_disp, exe_disp, ref_yaw, exe_yaw)]
    csv_path = out/'reference_execution_data.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    fig.subplots_adjust(left=.067, right=.985, bottom=.19, top=.77, wspace=.36)
    fields = [
        ('Forward displacement', r'$x-x_0$ (m)', ref_disp[:, 0], exe_disp[:, 0]),
        ('Lateral displacement', r'$y-y_0$ (cm)', 100*ref_disp[:, 1], 100*exe_disp[:, 1]),
        ('Body yaw', r'$\psi$ (deg)', np.rad2deg(ref_yaw), np.rad2deg(exe_yaw)),
    ]
    for index, (ax, (title, ylabel, reference, execution)) in enumerate(zip(axes, fields)):
        panel(ax, chr(97+index), title, 'Time (s)', ylabel)
        ax.plot(time, reference, color=BLUE, lw=2.2, ls='-', marker=None,
                label='Generated reference', zorder=2)
        ax.plot(time, execution, color=ORANGE, lw=1.6, ls='-', marker=None,
                label='Controller execution', zorder=3)
        ax.set_xlim(0, 6)
        ax.set_xticks([0, 2, 4, 6])
        ax.margins(y=.10)
    axes[0].set_ylim(-.025, .635)
    axes[0].text(.04, .93, 'At 6 s: 0.600 m / 0.585 m', transform=axes[0].transAxes,
                 fontsize=10, va='top', color='#555555')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53, 1.025),
               ncol=2, handlelength=2.7, fontsize=12)
    save(fig, 'reference_execution_comparison')

    sources = [ref_file, exe_file, report_file, video_file]
    metadata = dict(
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        selected_attempt=str(attempt.relative_to(ROOT)), selection=video['selection'],
        execution='Feedback controller plus native PD; reference-collection validation',
        learned_policy_inference=False, accepted=True, condition='Nominal simulation parameters',
        sample_rate_hz=50, samples=301, duration_s=6.0,
        coordinates='World directions, shared initial position; x/y plotted as displacement from t=0',
        quaternion_order='wxyz', yaw='atan2 from recorded quaternion; no heading alignment',
        smoothing='None; continuous solid lines join the original 50 Hz samples',
        metric_samples='300 post-control states; initial frame excluded from RMSE',
        metrics=metrics, data_file=csv_path.name, data_sha256=sha(csv_path))
    metadata_path = out/'reference_execution_metadata.json'
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    return dict(sources=sources, assets=[csv_path.name, metadata_path.name], metadata=metadata)
