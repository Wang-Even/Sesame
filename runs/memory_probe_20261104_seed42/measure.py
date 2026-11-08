"""100ms read-only NVML process-memory monitor and isolated short-run launcher."""
import csv
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parent


class Process(C.Structure):
    _fields_ = [('pid', C.c_uint), ('usedGpuMemory', C.c_ulonglong),
                ('gpuInstanceId', C.c_uint), ('computeInstanceId', C.c_uint)]


class Memory(C.Structure):
    _fields_ = [('total', C.c_ulonglong), ('free', C.c_ulonglong), ('used', C.c_ulonglong)]


def main():
    if (OUT/'result.json').exists():
        raise RuntimeError('Existing training result; refusing duplicate launch')
    nvml = C.CDLL('libnvidia-ml.so.1')
    nvml.nvmlErrorString.restype = C.c_char_p

    def check(code):
        if code:
            raise RuntimeError(nvml.nvmlErrorString(code).decode())

    check(nvml.nvmlInit_v2())
    handle = C.c_void_p()
    check(nvml.nvmlDeviceGetHandleByIndex_v2(0, C.byref(handle)))
    name = C.create_string_buffer(256)
    check(nvml.nvmlDeviceGetName(handle, name, len(name)))

    def snapshot():
        mem = Memory()
        check(nvml.nvmlDeviceGetMemoryInfo(handle, C.byref(mem)))
        by_pid = {}
        for fn_name, kind in [('nvmlDeviceGetComputeRunningProcesses_v2', 'compute'),
                              ('nvmlDeviceGetGraphicsRunningProcesses_v2', 'graphics')]:
            array = (Process * 128)()
            count = C.c_uint(128)
            check(getattr(nvml, fn_name)(handle, C.byref(count), array))
            for item in array[:count.value]:
                if item.usedGpuMemory == (2**64-1):
                    continue
                row = by_pid.setdefault(int(item.pid), dict(bytes=0, contexts=[]))
                row['bytes'] = max(row['bytes'], int(item.usedGpuMemory))
                row['contexts'].append(kind)
        return dict(device_used_bytes=mem.used, device_total_bytes=mem.total, processes=by_pid)

    before = snapshot()
    print('GPU_BEFORE=' + json.dumps(dict(gpu=name.value.decode(), **before)), flush=True)
    if before['device_total_bytes'] - before['device_used_bytes'] < 6 * 1024**3:
        raise RuntimeError('Less than 6 GiB free; did not launch')
    cmd = ['/home/ubuntu/miniforge3/bin/conda', 'run', '-n', 'env_isaaclab', '--no-capture-output',
           '/home/ubuntu/miniforge3/envs/env_isaaclab/bin/python', str(OUT/'profile_training.py'),
           '--num-envs', '256', '--iterations', '20', '--steps-per-env', '32',
           '--eval-episodes', '1', '--seed', '42', '--run-dir', str(OUT)]
    env = os.environ.copy()
    env.update(CONDA_NO_PLUGINS='true', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    record = dict(status='running', command=cmd, gpu=name.value.decode(),
                  sample_interval_requested_s=0.1, before=before, started_at_unix=time.time(),
                  scope='Headless baseline PPO, 256 environments, 20 updates, 32 steps/env; separate post-training evaluation.',
                  method='NVML per-process total GPU memory; compute/graphics contexts deduplicated by PID using max, not sum.',
                  files_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), OUT/'profile_training.py', Path('/home/ywwang/sesame/scripts/train_ppo.py')]})
    (OUT/'measurement.json').write_text(json.dumps(record, indent=2)+'\n')
    with (OUT/'train.log').open('w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, cwd='/home/ywwang/sesame', start_new_session=True)
    records = []
    own_pids = set()
    last_progress = 0.0
    with (OUT/'gpu_samples.jsonl').open('w') as samples:
        while True:
            tick = time.monotonic()
            stamp = time.time()
            phase = json.loads((OUT/'phase.json').read_text()) if (OUT/'phase.json').exists() else {}
            pid = phase.get('pid')
            if pid:
                own_pids.add(pid)
            snap = snapshot()
            target = snap['processes'].get(pid)
            row = dict(timestamp=stamp, phase=phase.get('phase', 'launcher'), training_pid=pid,
                       process_used_bytes=target['bytes'] if target else None,
                       contexts=target['contexts'] if target else [],
                       device_used_bytes=snap['device_used_bytes'],
                       other_processes={k:v for k,v in snap['processes'].items() if k != pid})
            samples.write(json.dumps(row)+'\n')
            samples.flush()
            records.append(row)
            if tick-last_progress > 15:
                print('SAMPLE=' + json.dumps({k:row[k] for k in ['phase','training_pid','process_used_bytes','device_used_bytes']}), flush=True)
                last_progress = tick
            if proc.poll() is not None:
                break
            time.sleep(max(0, 0.1-(time.monotonic()-tick)))
    result = json.loads((OUT/'result.json').read_text()) if (OUT/'result.json').exists() else {}
    peaks = {}
    for row in records:
        value = row['process_used_bytes']
        if value is not None:
            for key in ('overall', row['phase']):
                if key not in peaks or value > peaks[key]['bytes']:
                    peaks[key] = dict(bytes=value, MiB=value/1024**2, GiB=value/1024**3,
                                      timestamp=row['timestamp'], phase=row['phase'], pid=row['training_pid'])
    gaps = [b['timestamp']-a['timestamp'] for a,b in zip(records,records[1:])]
    record.update(status='completed' if proc.returncode==0 and result.get('status')=='completed' else 'failed',
                  exit_code=proc.returncode, training_status=result.get('status'),
                  training_pid=sorted(own_pids), samples=len(records),
                  effective_mean_sample_interval_s=sum(gaps)/len(gaps) if gaps else None,
                  maximum_sample_gap_s=max(gaps) if gaps else None,
                  peaks=peaks, after=snapshot(), finished_at_unix=time.time(),
                  completed_training_iterations=result.get('training_iterations_this_run'),
                  environment_transitions_total=result.get('environment_transitions_total'),
                  limits='Sampled maximum of this short run, not the historical 250-update peak. Sub-interval spikes may be missed.')
    if (OUT/'torch_memory.json').exists():
        record['pytorch'] = json.loads((OUT/'torch_memory.json').read_text())
    (OUT/'measurement.json').write_text(json.dumps(record,indent=2)+'\n')
    print('MEASUREMENT_RESULT=' + json.dumps(record), flush=True)
    check(nvml.nvmlShutdown())
    if record['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
