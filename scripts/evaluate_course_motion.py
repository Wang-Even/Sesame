"""Fresh CUDA-only motion diagnosis of four fixed final.pt checkpoints; no video.

Recorded states are pre-step frames 0..299 (0..5.98 s). The Evaluator still
scores 300 control transitions (6 s). Failed clips are truncated at their last
valid pre-step state; automatically reset states never enter motion statistics.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/supermimic_official_5efbf114/src')]
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'.runtime/research_cache'))
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('OMP_NUM_THREADS', '4')
GROUPS = ('ppo', 'supermimic', 'supermimic_deepseek', 'supermimic_codex')
STEPS, DT, SEED = 300, .02, 91042
CRITICAL_SOURCES = ('sesame_research/evaluation.py', 'sesame_research/task.py',
                    'sesame_research/robot.py', 'sesame_research/reward.py')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def preflight(suite, dataset=None):
    """File/configuration checks only; safe before CUDA imports."""
    suite_protocol = read(suite/'protocol.json')
    recorded_test = Path(suite_protocol['dataset'])/'common_test'
    test = recorded_test if dataset is None else dataset.resolve()/'common_test'
    if dataset is not None:
        protocol_path = test.parent/'protocol.json'
        if digest(protocol_path) != suite_protocol['dataset_protocol_sha256']:
            raise ValueError('Selected dataset differs from the training dataset')
        for relative, expected in read(protocol_path)['array_sha256'].items():
            if Path(relative).parts[0] == 'common_test' and digest(test.parent/relative) != expected:
                raise ValueError(f'Test data changed: {relative}')
    names = read(test/'motion_names.json')
    if read(test/'meta.json')['clips'] != 12 or len(names) != 12:
        raise ValueError('Expected 12 common test clips')
    if {source:sum(n.startswith(source+'__') for n in names)
        for source in ('original','deepseek','codex')} != {'original':4,'deepseek':4,'codex':4}:
        raise ValueError('Expected four test clips from each reference source')
    records = {}
    common = None
    for name in GROUPS:
        run = suite/name
        saved, status = read(run/'protocol.json'), read(run/'status.json')
        args = saved['args']
        if status['status'] != 'complete' or not (run/'final.pt').is_file():
            raise ValueError(f'{name}: completed final.pt required')
        if Path(args['test_file']).resolve() != recorded_test.resolve():
            raise ValueError(f'{name}: common test source mismatch')
        if args['eval_steps'] != STEPS or args['physics_dt'] != .0005:
            raise ValueError(f'{name}: require 300 steps and 0.5 ms physics')
        settings = dict(seed=args['seed'], physics_dt=args['physics_dt'],
                        diagnostic_only_metrics=args.get('diagnostic_only_metrics', []),
                        reward_version=saved['reward_version'])
        if common is not None and settings != common:
            raise ValueError(f'{name}: inconsistent evaluation settings')
        common = settings
        for relative in CRITICAL_SOURCES:
            if digest(ROOT/relative) != saved['sources'][relative]:
                raise ValueError(f'{name}: current source differs from training snapshot: {relative}')
        records[name] = dict(protocol=saved, final_status=status,
                             checkpoint=str((run/'final.pt').resolve()),
                             checkpoint_sha256=digest(run/'final.pt'),
                             training_protocol_sha256=digest(run/'protocol.json'))
    return suite_protocol, test, names, records, common


def motion_statistics(torch, trace, completed, names):
    """All trajectory arithmetic/reductions execute on CUDA."""
    alive = trace['alive']
    if not bool((alive[:-1] | ~alive[1:]).all()):
        raise ValueError('Alive masks must be continuous prefixes')
    counts = alive.sum(0)
    if not bool((counts >= 1).all()):
        raise ValueError('Missing initial states')
    ends = counts-1
    ids = torch.arange(len(names), device='cuda:0')
    positions, quaternions = trace['root_pos_w'], trace['root_quat_w']
    delta = positions-positions[0]
    net = delta[ends, ids]
    duration = ends.to(torch.float64)*DT
    speed_valid = duration > 0
    speed = net[:,0]/duration.clamp_min(DT)
    distance = net[:,:2].norm(dim=-1)
    direction_valid = distance > 1e-6
    direction = torch.rad2deg(torch.atan2(net[:,1], net[:,0]))
    w,x,y,z = quaternions.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    yaw_deg = torch.rad2deg(yaw)
    final_yaw = yaw_deg[ends, ids]
    yaw_change = yaw[ends, ids]-yaw[0]
    yaw_change = torch.rad2deg(torch.atan2(yaw_change.sin(), yaw_change.cos()))
    yaw_rms = torch.rad2deg((yaw.square()*alive).sum(0).div(counts).sqrt())
    max_lateral = torch.where(alive, delta[:,:,1].abs(), 0.).amax(0)
    reference_net = trace['reference_pos_w'][ends,ids]-trace['reference_pos_w'][0]
    reference_speed = reference_net[:,0]/duration.clamp_min(DT)
    vel = trace['root_lin_vel_w']
    mean_sampled_vx = (vel[:,:,0]*alive).sum(0)/counts
    data = dict(valid_pre_step_states=counts, statistics_duration_s=duration,
                forward_displacement_m=net[:,0], lateral_displacement_m=net[:,1],
                max_abs_lateral_displacement_m=max_lateral,
                net_planar_displacement_m=distance,
                forward_speed_from_displacement_m_s=speed,
                net_movement_direction_deg=direction,
                initial_body_yaw_deg=yaw_deg[0], final_body_yaw_deg=final_yaw,
                body_yaw_change_deg=yaw_change, world_body_yaw_rms_deg=yaw_rms,
                reference_forward_speed_from_displacement_m_s=reference_speed,
                mean_sampled_vx_m_s=mean_sampled_vx,
                completed_without_local_termination=completed)
    host = {key:value.cpu().tolist() for key,value in data.items()}
    valid_speed, valid_direction = speed_valid.cpu().tolist(), direction_valid.cpu().tolist()
    rows = []
    for i, name in enumerate(names):
        row = {'clip_index':i, 'clip':name, 'source':name.split('__',1)[0]}
        row.update({key:value[i] for key,value in host.items()})
        if not valid_speed[i]:
            row['forward_speed_from_displacement_m_s'] = None
            row['reference_forward_speed_from_displacement_m_s'] = None
        if not valid_direction[i]:
            row['net_movement_direction_deg'] = None
        rows.append(row)

    def aggregate(mask):
        def mean(value, valid=None):
            selected = mask if valid is None else mask & valid
            return float(value[selected].mean()) if bool(selected.any()) else None
        return dict(episodes=int(mask.sum()), complete_episodes=int(completed[mask].sum()),
                    valid_speed_episodes=int((mask & speed_valid).sum()),
                    direction_defined_episodes=int((mask & direction_valid).sum()),
                    mean_forward_speed_from_displacement_m_s=mean(speed, speed_valid),
                    mean_reference_forward_speed_from_displacement_m_s=mean(reference_speed, speed_valid),
                    mean_forward_displacement_m=mean(net[:,0]),
                    mean_abs_lateral_displacement_m=mean(net[:,1].abs()),
                    mean_max_abs_lateral_displacement_m=mean(max_lateral),
                    mean_abs_net_movement_direction_deg=mean(direction.abs(), direction_valid),
                    mean_abs_final_body_yaw_deg=mean(final_yaw.abs()),
                    mean_world_body_yaw_rms_deg=mean(yaw_rms))
    all_mask = torch.ones(len(names), device='cuda:0', dtype=torch.bool)
    summary = {'all':aggregate(all_mask), 'by_source':{}}
    for source in ('original','deepseek','codex'):
        mask = torch.tensor([n.startswith(source+'__') for n in names],device='cuda:0',dtype=torch.bool)
        summary['by_source'][source] = aggregate(mask)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=ROOT/'runs/course_comparison_20261115')
    parser.add_argument('--dataset', type=Path, help='Local copy of the original dataset, verified against the training protocol')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/course_motion_evaluation_20261115')
    parser.add_argument('--check-config', action='store_true', help='Only read completed-run metadata; no CUDA or model computation')
    args = parser.parse_args()
    suite, output = args.suite.resolve(), args.output.resolve()
    _, test, names, records, common = preflight(suite, args.dataset)
    if args.check_config:
        print(json.dumps(dict(config_valid=True, groups=GROUPS, test=str(test), clips=12,
                              gpu_or_model_used=False))); return
    if output.exists():
        raise FileExistsError('Choose a fresh output directory; training results are never overwritten')
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    torch.cuda.set_device(0); torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.cuda.set_per_process_memory_fraction(.40)
    from sesame_research.evaluation import Evaluator
    from sesame_research.reward import REWARD_VERSION
    from supermimic.runners.run_config import load_runner_cfg
    from supermimic.runners.supermimic_runner import build_actor, build_world_model
    if common['reward_version'] != REWARD_VERSION:
        raise ValueError('Current reward version differs from these checkpoints')
    output.mkdir(parents=True)
    source_hashes = {}
    for relative in (str(Path(__file__).relative_to(ROOT)), *CRITICAL_SOURCES):
        source = ROOT/relative; destination = output/'sources'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(source.read_bytes()); source_hashes[relative] = digest(source)
    protocol = dict(suite=str(suite), suite_protocol_sha256=digest(suite/'protocol.json'),
                    seed=SEED, groups=GROUPS, test_source=str(test), clip_order=names,
                    test_files_sha256={p.name:digest(p) for p in test.iterdir() if p.is_file()},
                    checkpoint_selection='Fixed final.pt for all groups; all test clips in stored order, one evaluation each',
                    evaluation='Fresh unified motion diagnosis, separate from the training-end final_test results',
                    course_scoring=False, video_rendering=False,
                    physics_dt_s=.0005, control_dt_s=DT, evaluation_steps=STEPS,
                    reward_horizon_s=STEPS*DT, statistics_interval_s=[0.,(STEPS-1)*DT],
                    early_failure='Motion statistics stop at the last alive pre-step state; reset states are retained only in raw trace with alive=False',
                    complete_definition='No local failure during all 300 transitions, including the last transition; normal timeout does not count as failure',
                    direction_definition='atan2(net world Y displacement, net world X displacement), separate from world-frame body yaw; undefined below 1e-6 m net displacement',
                    speed_definition='Net world X displacement divided by valid pre-step trace duration; 5.98 s for full clips. Raw instantaneous velocity is diagnostic only.',
                    aggregation='Means of per-clip CUDA statistics; early-failed clips use their valid durations, complete count reported separately',
                    initial_state_match_tolerance=1e-6,
                    reward_version=REWARD_VERSION, source_sha256=source_hashes, models={})
    write(output/'protocol.json', protocol)
    state = dict(status='running', completed=[], started_unix=time.time())
    write(output/'status.json',state)
    all_rows, comparison, baseline_initial = [], [], None
    try:
        for group in GROUPS:
            evaluator = None
            state['current'] = group; write(output/'status.json',state)
            folder = output/group; folder.mkdir()
            saved = records[group]['protocol']
            try:
                torch.manual_seed(saved['args']['seed'])
                evaluator = Evaluator(test, SEED, .0005, diagnostic_only_metrics=common['diagnostic_only_metrics'])
                cfg = load_runner_cfg({'runner_cfg':saved['runner_config']})
                wm = build_world_model(cfg,evaluator.env,'cuda:0')
                actor = build_actor(cfg,wm,'cuda:0')
                bundle = torch.load(records[group]['checkpoint'],map_location='cuda:0',weights_only=False)
                if 'world_model' in bundle:
                    wm.load_bundle(bundle['world_model'])
                actor.load_state_dict(bundle['actor'],strict=True); actor.eval()
                store = {}
                def capture(t,raw,alive,reference):
                    d = raw.scene['robot'].data
                    values = dict(qpos=raw.sim.data.qpos,qvel=raw.sim.data.qvel,
                        root_pos_w=d.root_link_pos_w-raw.scene.env_origins,
                        root_quat_w=d.root_link_quat_w,root_lin_vel_w=d.root_link_lin_vel_w,
                        root_ang_vel_w=d.root_link_ang_vel_w,joint_pos=d.joint_pos,joint_vel=d.joint_vel,
                        reference_pos_w=reference[0][:,0],reference_quat_w=reference[1][:,0],
                        reference_lin_vel_w=reference[2][:,0],alive=alive)
                    for key,value in values.items(): store.setdefault(key,[]).append(value.clone())
                metrics = evaluator.run(actor,wm.representation,wm.sim_context,cfg.history_length,STEPS,trace_callback=capture)
                trace = {key:torch.stack(values) for key,values in store.items()}
                assert all(len(value)==STEPS for value in trace.values())
                if not all(bool(torch.isfinite(value).all()) for value in trace.values()):
                    raise ValueError('Nonfinite recorded states')
                last_failure = evaluator.env.unwrapped.termination_manager.terminated.clone()
                completed = trace['alive'].all(0) & ~last_failure
                if abs(float((~completed).float().mean())-metrics['fall_rate']) > 1e-6:
                    raise ValueError('Completion mask disagrees with Evaluator termination fraction')
                initial = torch.cat([trace[k][0].flatten() for k in ('root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','joint_pos','joint_vel')])
                if baseline_initial is None: baseline_initial = initial.clone()
                initial_error = float((initial-baseline_initial).abs().max())
                if initial_error > 1e-6: raise ValueError('Initial states differ between groups')
                per_clip, summary = motion_statistics(torch,trace,completed,names)
                episode_normalized_returns = (torch.tensor(metrics['episode_returns'],device='cuda:0')/(STEPS*DT*5.)).cpu().tolist()
                for i,row in enumerate(per_clip):
                    row.update(group=group, trajectory_return=metrics['episode_returns'][i],
                               normalized_return=episode_normalized_returns[i],
                               **metrics['episode_metrics'][i])
                np.savez_compressed(folder/'policy_trace.npz',
                    **{key:value.cpu().numpy() for key,value in trace.items()},
                    last_transition_terminated=last_failure.cpu().numpy(),
                    completed_without_local_termination=completed.cpu().numpy(),
                    sample_times_s=(torch.arange(STEPS,device='cuda:0')*DT).cpu().numpy(),fps=50)
                write(folder/'evaluation.json',metrics); write(folder/'motion_statistics.json',summary)
                csv_write(folder/'per_clip.csv',per_clip); all_rows.extend(per_clip)
                comparison.append(dict(group=group, return_value=metrics['return'],
                    normalized_return=metrics['normalized_return'], heading_rms_deg=metrics['heading_rms_deg'],
                    velocity_rmse_m_s=metrics['velocity_rmse_m_s'],
                    root_position_rmse_m=metrics['root_position_rmse_m'], **summary['all']))
                model_protocol = dict(checkpoint=records[group]['checkpoint'],
                    checkpoint_sha256=records[group]['checkpoint_sha256'],
                    checkpoint_real_steps=bundle['steps'],
                    training_protocol_sha256=records[group]['training_protocol_sha256'],
                    trace_sha256=digest(folder/'policy_trace.npz'),
                    initial_state_max_abs_difference_from_first_group=initial_error,
                    world_model_rollouts_used=False,
                    inference='Actor outputs executed in native CUDA physics; world model only supplies representation/configuration interfaces')
                write(folder/'protocol.json',model_protocol)
                protocol['models'][group]=model_protocol; write(output/'protocol.json',protocol)
                state['completed'].append(group); write(output/'status.json',state)
                print(json.dumps(comparison[-1]),flush=True)
            finally:
                if evaluator is not None: evaluator.close()
            del evaluator,actor,wm,bundle,store,trace,initial,last_failure,completed
            gc.collect(); torch.cuda.empty_cache()
        csv_write(output/'per_clip.csv',all_rows); csv_write(output/'comparison.csv',comparison)
        write(output/'comparison.json',comparison)
        state.update(status='complete',finished_unix=time.time()); write(output/'status.json',state)
    except BaseException:
        state.update(status='failed',finished_unix=time.time(),error=traceback.format_exc())
        write(output/'status.json',state); raise


if __name__ == '__main__':
    main()
