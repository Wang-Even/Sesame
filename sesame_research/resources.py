"""Whole-process GPU memory sampling without interfering with other jobs."""
import csv
import os
import subprocess
import threading
import time


class GpuMemoryMonitor:
    def __init__(self, path, limit_mib=14336):
        self.path = path
        self.limit_mib = limit_mib
        self.peak_mib = 0
        self.exceeded = False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run,daemon=True)
        self.thread.start()

    def _run(self):
        with self.path.open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["unix_s","pid","process_memory_mib","gpu_used_mib","gpu_utilization_pct"])
            while not self.stop.is_set():
                try:
                    result = subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=3,check=True)
                    gpu=subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=3,check=True)
                    gpu_used,gpu_util=[int(v.strip()) for v in gpu.stdout.splitlines()[0].split(",")]
                    for line in result.stdout.splitlines():
                        pid, memory = [int(x.strip()) for x in line.split(",")]
                        if pid == os.getpid():
                            self.peak_mib = max(self.peak_mib,memory)
                            self.exceeded |= memory > self.limit_mib
                            writer.writerow([time.time(),pid,memory,gpu_used,gpu_util]); f.flush()
                except (subprocess.SubprocessError, ValueError):
                    pass
                self.stop.wait(1.)

    def check(self):
        if self.exceeded:
            raise RuntimeError(f"Own-process memory exceeded budget: {self.peak_mib} MiB")

    def close(self):
        self.stop.set(); self.thread.join(timeout=4)
