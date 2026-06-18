from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from app.core import sam3_workers


def test_worker_specs_use_configured_gpus_and_raw_annotation_flags(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(sam3_workers, "SAM3_CHECKPOINT", "/models/sam3.1.pt")
    job_dir = tmp_path / "job"
    done_file = job_dir / "sam3" / "frame_prep.done"
    specs = sam3_workers.build_worker_specs(
        job_dir,
        {
            "allowed_gpus": [2, 4],
            "model_python": "/opt/sam3/bin/python",
            "sam3_repo": "/opt/sam3",
            "score_threshold": 0.4,
            "save_masks": True,
        },
        done_file,
    )

    assert [spec.gpu for spec in specs] == [2, 4]
    assert specs[0].command[0] == "/opt/sam3/bin/python"
    assert "--frames-db" in specs[0].command
    assert "--sam3-repo" in specs[0].command
    assert "--checkpoint" in specs[0].command
    assert "--save-masks" in specs[0].command
    assert specs[0].command[specs[0].command.index("--dtype") + 1] == "bf16"
    assert "--nms-iou-threshold" not in specs[0].command
    assert str(done_file) in specs[0].command


def test_worker_specs_can_skip_masks(tmp_path: Path) -> None:
    specs = sam3_workers.build_worker_specs(
        tmp_path / "job",
        {"allowed_gpus": [0], "save_masks": False},
        tmp_path / "job" / "sam3" / "frame_prep.done",
    )

    assert "--save-masks" not in specs[0].command


def test_selected_gpus_require_max_15gb_or_half_total_memory(monkeypatch) -> None:
    def fake_check_output(*_args, **_kwargs):
        return "\n".join(
            [
                "0, 34254, 49152",  # pass: >= 24GB
                "1, 23000, 49152",  # fail: below half capacity
                "2, 18000, 24576",  # pass: >= 15GB
                "3, 12000, 24576",  # fail: below 15GB
            ]
        )

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert sam3_workers.selected_gpus() == [0, 2]
    assert sam3_workers.selected_gpus([0, 1, 2, 3]) == [0, 2]


def test_selected_gpus_auto_selects_least_occupied_count(monkeypatch) -> None:
    def fake_check_output(*_args, **_kwargs):
        return "\n".join(
            [
                "0, 32000, 49152",
                "1, 44000, 49152",
                "2, 18000, 24576",
                "3, 12000, 24576",  # fail: below 15GB
            ]
        )

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert sam3_workers.selected_gpus(count=2) == [1, 0]


def test_build_worker_specs_uses_gpu_count_when_ids_are_blank(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_check_output(*_args, **_kwargs):
        return "\n".join(
            [
                "0, 30000, 49152",
                "1, 46000, 49152",
                "2, 42000, 49152",
            ]
        )

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    specs = sam3_workers.build_worker_specs(
        tmp_path / "job",
        {"gpu_count": 2, "save_masks": False},
        tmp_path / "job" / "sam3" / "frame_prep.done",
    )

    assert [spec.gpu for spec in specs] == [1, 2]


def test_selected_gpus_pass_configured_ids_when_nvidia_smi_is_unavailable(monkeypatch) -> None:
    def fake_check_output(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert sam3_workers.selected_gpus([6, 7]) == [6, 7]


def test_build_worker_specs_fails_when_configured_gpus_are_too_busy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_check_output(*_args, **_kwargs):
        return "0, 1000, 49152\n1, 2000, 49152\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    with pytest.raises(RuntimeError, match="enough free memory"):
        sam3_workers.build_worker_specs(
            tmp_path / "job",
            {"allowed_gpus": [0, 1]},
            tmp_path / "job" / "sam3" / "frame_prep.done",
        )
