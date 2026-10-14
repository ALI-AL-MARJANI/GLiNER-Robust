"""Run metadata written next to every result file (``env`` block of each JSON)."""

from __future__ import annotations

import os
import sys
import platform
import subprocess
import importlib.metadata
from typing import Any, Dict, Optional
from datetime import datetime, timezone

_PACKAGES = (
    "torch",
    "transformers",
    "gliner",
    "onnx",
    "onnxruntime",
    "openvino",
    "nncf",
    "numpy",
    "accelerate",
)


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_model() -> str:
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False)
        if out.stdout.strip():
            return out.stdout.strip()
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_gb() -> Optional[float]:
    try:
        import psutil  # noqa: PLC0415

        return round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        return None


def _gpu() -> Optional[str]:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if torch.backends.mps.is_available():
            return "Apple MPS"
    except ImportError:
        pass
    return None


def collect_env() -> Dict[str, Any]:
    """Versions, hardware and git state of the current process."""
    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": _gpu(),
        "packages": versions,
    }
