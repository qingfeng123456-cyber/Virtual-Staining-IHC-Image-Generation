"""Low-overhead NVIDIA utilization sampling without an extra dependency."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

_QUERY_FIELDS = (
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "power.draw",
    "temperature.gpu",
)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


@dataclass
class NvidiaSmiMonitor:
    """Sample one GPU in a daemon thread and return epoch-level summaries."""

    enabled: bool = False
    interval_seconds: float = 2.0
    device_index: int = 0
    _samples: list[tuple[float, ...]] = field(default_factory=list, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _executable: str | None = field(default=None, init=False)

    def start(self) -> None:
        if not self.enabled:
            return
        self._executable = shutil.which("nvidia-smi")
        if self._executable is None:
            return
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="virtual-staining-gpu-monitor",
            daemon=True,
        )
        self._thread.start()

    def _sample(self) -> None:
        if self._executable is None:
            return
        creation_flags = (
            subprocess.CREATE_NO_WINDOW
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        )
        result = subprocess.run(
            [
                self._executable,
                f"--query-gpu={','.join(_QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(2.0, self.interval_seconds),
            creationflags=creation_flags,
        )
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 0 or not rows:
            return
        row = rows[min(max(0, self.device_index), len(rows) - 1)]
        values = tuple(float(value.strip()) for value in row.split(","))
        if len(values) == len(_QUERY_FIELDS):
            self._samples.append(values)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except (OSError, subprocess.SubprocessError, ValueError):
                # Monitoring is diagnostic only; a transient nvidia-smi
                # failure must not interrupt training.
                if self._stop.is_set():
                    break
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, Any]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(3.0, self.interval_seconds + 1.0))
            self._thread = None
        if not self._samples:
            return {"gpu_monitor/sample_count": 0}
        columns = list(zip(*self._samples, strict=True))
        names = ("gpu_util_percent", "memory_util_percent", "memory_used_mib", "power_w", "temperature_c")
        report: dict[str, Any] = {"gpu_monitor/sample_count": len(self._samples)}
        for name, column in zip(names, columns, strict=True):
            values = list(column)
            report[f"gpu_monitor/{name}_mean"] = float(mean(values))
            report[f"gpu_monitor/{name}_p95"] = _percentile(values, 0.95)
            report[f"gpu_monitor/{name}_max"] = float(max(values))
        return report
