"""Tests for device resolution, run metadata and output-directory protection."""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_utils
from run_utils import (
    environment_metadata, find_checkpoint, prepare_out_dir, resolve_device,
)


def _availability(monkeypatch, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize("cuda, mps, expected", [
    (True,  True,  "cuda"),
    (True,  False, "cuda"),
    (False, True,  "mps"),
    (False, False, "cpu"),
])
def test_auto_priority_is_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    _availability(monkeypatch, cuda, mps)
    assert resolve_device("auto") == expected


def test_cpu_is_always_honoured(monkeypatch):
    _availability(monkeypatch, True, True)
    assert resolve_device("cpu") == "cpu"


@pytest.mark.parametrize("requested", ["cuda", "mps"])
def test_unavailable_accelerator_fails_instead_of_falling_back(monkeypatch, requested):
    _availability(monkeypatch, False, False)
    with pytest.raises(RuntimeError, match=requested):
        resolve_device(requested)


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError):
        resolve_device("gpu")


def test_metadata_is_json_serialisable_and_complete():
    meta = environment_metadata("auto", resolve_device("auto"))
    json.dumps(meta)  # plain Python values only, no tensors or devices
    for key in ("requested_device", "resolved_device", "torch_version", "python_version",
                "platform", "system", "machine", "processor", "cuda_available",
                "cuda_version", "gpu_name", "mps_available", "git_commit", "timestamp_utc"):
        assert key in meta


def test_out_dir_with_previous_run_is_refused(tmp_path):
    run = tmp_path / "run"
    assert prepare_out_dir(str(run)) == str(run)
    prepare_out_dir(str(run))  # an empty run directory may be reused
    (run / "loss_curve.png").write_bytes(b"earlier result")
    with pytest.raises(FileExistsError, match="loss_curve.png"):
        prepare_out_dir(str(run))
    assert (run / "loss_curve.png").read_bytes() == b"earlier result"


def test_find_checkpoint_prefers_newest_run_then_legacy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        find_checkpoint()
    (tmp_path / "checkpoint.pt").write_bytes(b"legacy")
    assert find_checkpoint() == "checkpoint.pt"
    for i, name in enumerate(["a", "b"]):
        d = tmp_path / "results" / name
        d.mkdir(parents=True)
        (d / "checkpoint.pt").write_bytes(b"run")
        os.utime(d / "checkpoint.pt", (1_000 + i, 1_000 + i))
    assert find_checkpoint() == os.path.join("results", "b", "checkpoint.pt")


def test_run_files_cover_every_output():
    assert set(run_utils.RUN_FILES) == {"checkpoint.pt", "loss_curve.png", "run_meta.json"}
