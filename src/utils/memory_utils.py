# src/utils/memory_utils.py

from typing import Dict, Any
import torch


def get_gpu_memory_info(device: torch.device) -> Dict[str, float]:
    """Returns allocated, reserved, and peak GPU memory in MB if CUDA is available."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}

    return {
        "gpu_allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "gpu_reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "gpu_peak_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }


def log_memory_usage(prefix: str = "", device: torch.device = None):
    """Prints GPU memory usage to standard output."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info = get_gpu_memory_info(device)
    if info:
        print(
            f"[{prefix}] GPU Alloc: {info['gpu_allocated_mb']:.1f}MB | "
            f"Res: {info['gpu_reserved_mb']:.1f}MB | Peak: {info['gpu_peak_mb']:.1f}MB"
        )
