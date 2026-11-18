#!/usr/bin/env python3
"""Package recorded arrays and fixed splits; file/metadata work only, no models."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--supplement', type=Path, action='append', default=[], help='Completed collection directory inside dataset; repeatable')
    args = parser.parse_args()
    root = args.dataset.resolve()
    result = json.loads((root/'result.json').read_text())
    if result['status'] != 'completed': raise RuntimeError('Collection is not complete')
    protocol = json.loads((root/'protocol.json').read_text())
    if protocol['smoke']: raise RuntimeError('Do not package a smoke run as training data')
    rows = json.loads((root/'episode_index.json').read_text())
    supplemental_profiles = []
    for supplemental in args.supplement:
        supplement = supplemental.resolve()
        prefix = supplement.relative_to(root)
        if json.loads((supplement/'result.json').read_text())['status'] != 'completed':
            raise RuntimeError('Supplement is incomplete')
        supplemental_protocol = json.loads((supplement/'protocol.json').read_text())
        supplemental_profiles.extend(supplemental_protocol['profiles'])
        for row in json.loads((supplement/'episode_index.json').read_text()):
            row = dict(row)
            row['original_id'] = row['id']
            row['id'] = prefix.name + '_' + row['id']
            for field in ('path','metadata_path'): row[field] = str(prefix / row[field])
            row['collection_protocol'] = str(prefix / 'protocol.json')
            rows.append(row)
    save(root/'combined_episode_index.json', rows)
    references = []
    split_summary = {}
    for split in ('train', 'validation', 'test'):
        group = [r for r in rows if r['split'] == split]
        usable = [r for r in group if r['world_model_usable']]
        rejected = [r for r in group if not r['world_model_usable']]
        selected = []
        for profile in ('straight_010', 'straight_015', 'straight_020'):
            candidates = [r for r in group if r['profile'] == profile and not r['reference_rejection_reasons']]
            if split == 'train':
                candidates.sort(key=lambda r:(r['metrics']['target_acceleration_rms_rad_s2'],r['id']))
            else:
                candidates.sort(key=lambda r:r['id'])
            selected.extend(candidates[:protocol['reference_caps_per_speed'][split]])
        for row in selected:
            source = root/row['path']
            if digest(source) != row['sha256']: raise RuntimeError('Changed episode data')
            start = protocol['reference_gates']['warmup_steps']
            with np.load(source, allow_pickle=False) as data:
                robot = data['state_root_state'][start:]
                bodies = data['state_body_link_state'][start:]
                # Slicing/copying archived arrays is file conversion, not model computation.
                motion = {'fps': np.asarray(50, dtype=np.int32),
                          'root_pos': robot[:, :3], 'root_quat': robot[:, 3:7],
                          'root_com_lin_vel': robot[:, 7:10], 'root_ang_vel': robot[:, 10:13],
                          'joint_pos': data['state_joint_pos'][start:], 'joint_vel': data['state_joint_vel'][start:],
                          'body_pos': bodies[..., :3], 'body_quat': bodies[..., 3:7],
                          'body_lin_vel': bodies[..., 7:10], 'body_ang_vel': bodies[..., 10:13],
                          'foot_pos': data['state_foot_pos'][start:], 'foot_vel': data['state_foot_vel'][start:],
                          'body_contact_force': data['body_contact_force'][start-1:],
                          'joint_torque_estimate': data['joint_torque_estimate'][start-1:]}
                path = root/'references'/split/(row['id']+'.npz')
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    with np.load(path, allow_pickle=False) as old:
                        if set(old.files) != set(motion): raise RuntimeError('Existing reference schema differs')
                        for key, value in motion.items():
                            if old[key].tobytes() != value.tobytes(): raise RuntimeError('Existing reference differs')
                else:
                    np.savez_compressed(path, **motion)
            reference = {'id':row['id'], 'split':split, 'profile':row['profile'], 'policy_id':row['policy_id'],
                         'condition':row['condition'], 'source_episode':row['path'], 'source_sha256':row['sha256'],
                         'source_state_start_index':start, 'frames':row['frames']-start,
                         'duration_s':(row['frames']-start-1)*protocol['control_dt_s'],
                         'path':str(path.relative_to(root)), 'sha256':digest(path), 'metrics':row['metrics'],
                         'loopable':False, 'cyclic_boundary':'not validated; do not wrap last to first frame',
                         'quality':'passes fixed engineering gates; not an optimal expert or hardware validation'}
            save(path.with_suffix('.json'),reference)
            reference['metadata_path']=str(path.with_suffix('.json').relative_to(root))
            reference['metadata_sha256']=digest(path.with_suffix('.json'))
            references.append(reference)
        save(root/'splits'/(split+'.json'), {
            'split':split, 'episodes':[{'path':r['path'],'metadata_path':r['metadata_path'],'sha256':r['sha256'],
                                      'metadata_sha256':r['metadata_sha256'],'transitions':r['transitions'],'id':r['id']} for r in usable],
            'references':[r['path'] for r in references if r['split']==split],
            'excluded_episode_ids':[r['id'] for r in rejected],
            'normalization':'Fit normalization statistics on TRAIN only; keep validation/test read-only for evaluation.',
            'window_rule':'Sample contiguous windows inside one episode. Last terminal state is included; never concatenate resets.'})
        split_summary[split]={'episodes':len(group),'usable_episodes':len(usable),
                              'transitions':sum(r['transitions'] for r in usable),
                              'references':sum(r['split']==split for r in references),
                              'falls':sum(r['metrics']['fell'] for r in group),
                              'reference_profiles':dict(Counter(r['profile'] for r in references if r['split']==split))}
    save(root/'reference_index.json', references)
    # Preserve historical examples separately. They lack full Markov/controller
    # context and must never silently enter the new world-model training set.
    legacy=[]
    for path in sorted((ROOT/'runs/offpolicy_heading_20261107/evaluation/trajectories').glob('*.json')):
        destination=root/'legacy'/path.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and digest(destination)!=digest(path): raise RuntimeError('Legacy copy differs')
        if not destination.exists(): shutil.copy2(path,destination)
        source=json.loads(path.read_text())
        legacy.append({'path':str(destination.relative_to(root)),'source':str(path.relative_to(ROOT)),
                       'sha256':digest(destination),'frames':len(source['samples']),
                       'training_use':False,'reason':'Historical diagnostic example; incomplete state/controller/contact context.'})
    save(root/'legacy/index.json',legacy)
    profile_summary=[]
    profiles = list(dict.fromkeys(list(protocol['profiles']) + supplemental_profiles))
    for profile in profiles:
        group=[r for r in rows if r['profile']==profile]
        profile_summary.append({'profile':profile,'episodes':len(group),
                                'falls':sum(r['metrics']['fell'] for r in group),
                                'reference_eligible':sum(not r['reference_rejection_reasons'] for r in group),
                                'reference_selected':sum(r['profile']==profile for r in references),
                                'rejections':dict(Counter(x for r in group for x in r['reference_rejection_reasons']))})
    summary={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'packaged','validation_file':'validation.json',
             'splits':split_summary,'profiles':profile_summary,'references':len(references),
             'reference_duration_s':sum(r['duration_s'] for r in references),
             'episodes':len(rows),'transitions':sum(r['transitions'] for r in rows),
             'simulated_seconds':sum(r['transitions'] for r in rows)*protocol['control_dt_s'],
             'legacy_episodes':len(legacy),'legacy_frames':sum(r['frames'] for r in legacy),
             'unavailable_skills':['turning','hardware demonstrations'],
             'scope':'Initial Sesame adaptation data. Includes in-distribution commands and out-of-distribution probes; no SuperMimic policy has been trained.'}
    save(root/'summary.json',summary)
    lines=['# Sesame 动作与仿真数据','',
           '用于 SuperMimic 在 Sesame 上的适配实验。包含参考动作和仿真状态转移；不是官方人形动作数据集。','',
           '| 划分 | 可用回合 | 状态转移 | 参考轨迹 | 跌倒回合 |','| --- | ---: | ---: | ---: | ---: |']
    for split,s in split_summary.items():
        lines.append(f'| {split} | {s["usable_episodes"]} | {s["transitions"]} | {s["references"]} | {s["falls"]} |')
    lines.extend(['','参考轨迹为去掉前 1 秒后的连续 9 秒片段，50 Hz；保留真实速度，不做时间缩放或循环拼接。',
                  '完整字段和坐标约定见 `schema.json`；阈值与采集设置见 `protocol.json`；来源和哈希见各索引。',
                  '按完整回合与采样种子划分；同一物理初态的 PPO/SAC 配对保持在相同划分。只用 train 拟合归一化。',
                  '同一类步态和相同来源策略跨划分共享，因此测试主要检查初态/动力学泛化，不代表全新技能泛化。','',
                  '`episodes/` 含 T+1 帧状态、T 步动作/指令/终止标记；`references/` 是筛选后的参考动作。',
                  '`legacy/` 的 18 条历史轨迹仅供格式参考，不进入新训练集。',
                  '`legacy_reward` 沿用旧环境的固定 0.10 m/s 奖励，不是变速任务评分或模仿奖励；训练时须按参考动作重算。',
                  '接触力是每个控制周期末的刚体法向接触力，不能直接称为足尖接触；关节力矩为隐式驱动估计。',
                  '当前参考动作仅通过预设速度、航向、平滑性和关节限位筛选，不代表最优步态。启停、扰动轨迹作为动力学数据保存。','',
                  '使用方法见项目 `docs/SUPERMIMIC_DATA_20261110.md`。'])
    (root/'DATASET.md').write_text('\n'.join(lines)+'\n')
    for name in ('scripts/prepare_supermimic_dataset.py','scripts/validate_supermimic_dataset.py','sesame_rl/motion_dataset.py'):
        target=root/'processing_sources'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    tracked=[]
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.npz','.json','.py','.md') and path.name not in ('checksums.json','validation.json'):
            tracked.append({'path':str(path.relative_to(root)), 'bytes':path.stat().st_size,'sha256':digest(path)})
    save(root/'checksums.json',tracked)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
