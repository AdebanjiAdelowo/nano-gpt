"""
Device selection, run provenance, and output-directory helpers shared by
train.py and generate.py.
"""

import glob
import os
import platform
import subprocess
from datetime import datetime, timezone

import torch

DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")
RUN_FILES = ("checkpoint.pt", "loss_curve.png", "run_meta.json")


def resolve_device(requested: str = "auto") -> str:
    """
    Map a requested device to the one actually used.

    "auto" picks CUDA, then MPS, then CPU. An explicitly requested
    accelerator that is not available raises instead of falling back, so a
    run can never report one device while using another.
    """
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device '{requested}'; choose from {DEVICE_CHOICES}")

    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()

    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if requested == "cuda" and not cuda:
        raise RuntimeError("--device cuda was requested but CUDA is not available on this machine")
    if requested == "mps" and not mps:
        raise RuntimeError("--device mps was requested but MPS is not available on this machine")
    return requested


def git_commit() -> dict:
    """Commit hash and dirty flag of the repository holding this file, if any."""
    here = os.path.dirname(os.path.abspath(__file__))

    def _git(*args: str):
        try:
            out = subprocess.run(
                ["git", *args], cwd=here, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def environment_metadata(requested_device: str, resolved_device: str) -> dict:
    """Plain-Python (JSON-serialisable) description of the machine and software."""
    cuda = torch.cuda.is_available()
    git = git_commit()
    return {
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested_device": requested_device,
        "resolved_device":  resolved_device,
        "torch_version":    torch.__version__,
        "python_version":   platform.python_version(),
        "platform":         platform.platform(),
        "system":           platform.system(),
        "machine":          platform.machine(),
        "processor":        platform.processor() or None,
        "cuda_available":   cuda,
        "cuda_version":     torch.version.cuda,
        "gpu_name":         torch.cuda.get_device_name(0) if cuda else None,
        "mps_available":    torch.backends.mps.is_available(),
        "git_commit":       git["commit"],
        "git_dirty":        git["dirty"],
    }


def default_out_dir(resolved_device: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join("results", f"{stamp}-{resolved_device}")


def prepare_out_dir(out_dir: str) -> str:
    """
    Create the run directory, refusing to reuse one that already holds run
    outputs, so an earlier result is never overwritten.
    """
    existing = [f for f in RUN_FILES if os.path.exists(os.path.join(out_dir, f))]
    if existing:
        raise FileExistsError(
            f"{out_dir!r} already contains {', '.join(existing)}; "
            "choose a different --out-dir to keep the earlier run intact"
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def find_checkpoint(results_dir: str = "results", legacy: str = "checkpoint.pt") -> str:
    """Newest results/<run>/checkpoint.pt, else the legacy top-level checkpoint."""
    runs = glob.glob(os.path.join(results_dir, "*", "checkpoint.pt"))
    if runs:
        return max(runs, key=os.path.getmtime)
    if os.path.exists(legacy):
        return legacy
    raise FileNotFoundError(
        "No checkpoint found; run train.py first or pass --checkpoint <path>"
    )
