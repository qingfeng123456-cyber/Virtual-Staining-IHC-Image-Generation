"""Runtime environment inspection and hardware-aware defaults."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

import psutil
import torch


def environment_report() -> dict[str, Any]:
    """Return truthful Python, memory, PyTorch, CUDA, and GPU information."""

    memory = psutil.virtual_memory()
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "executable": os.path.abspath(os.sys.executable),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_total_gib": round(memory.total / 1024**3, 3),
        "ram_available_gib": round(memory.available / 1024**3, 3),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        report.update(
            {
                "gpu_index": index,
                "gpu_name": props.name,
                "gpu_compute_capability": list(torch.cuda.get_device_capability(index)),
                "gpu_total_mib": round(total / 1024**2),
                "gpu_free_mib": round(free / 1024**2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "amp_supported": hasattr(torch, "amp"),
            }
        )
    return report


def save_environment_report(path: str | Path) -> dict[str, Any]:
    """Save and return the environment report."""

    report = environment_report()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda without silently accepting unavailable CUDA."""

    lowered = requested.lower()
    if lowered == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(lowered)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def hardware_profile() -> dict[str, Any]:
    """Return conservative local training defaults from total GPU memory."""

    if not torch.cuda.is_available():
        return {"name": "cpu", "base_channels": 16, "batch_size": 1, "gradient_accumulation": 1}
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gib <= 8.25:
        return {"name": "gpu_8gb", "base_channels": 32, "batch_size": 2, "gradient_accumulation": 4}
    if total_gib < 20:
        return {"name": "gpu_16gb", "base_channels": 48, "batch_size": 4, "gradient_accumulation": 2}
    return {"name": "gpu_20gb_plus", "base_channels": 64, "batch_size": 8, "gradient_accumulation": 1}

