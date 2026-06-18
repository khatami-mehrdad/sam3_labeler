from __future__ import annotations

import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CAFE_CONSECUTIVE = ROOT / "tests" / "fixtures" / "cafe_consecutive"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--cafe-frames",
        type=int,
        default=int(os.environ.get("DGL_TEST_CAFE_FRAMES", "10")),
        help="Number of consecutive cafe frames to use per camera.",
    )


@pytest.fixture(scope="session")
def cafe_consecutive_dir() -> Path:
    return CAFE_CONSECUTIVE


@pytest.fixture(scope="session")
def cafe_camera_dirs(cafe_consecutive_dir: Path) -> list[Path]:
    return sorted(path for path in cafe_consecutive_dir.iterdir() if path.is_dir())


@pytest.fixture(scope="session")
def cafe_frame_paths(
    cafe_camera_dirs: list[Path],
    pytestconfig: pytest.Config,
) -> list[Path]:
    per_camera = pytestconfig.getoption("--cafe-frames")
    paths: list[Path] = []
    for camera_dir in cafe_camera_dirs:
        paths.extend(sorted(camera_dir.glob("*.jpg"))[:per_camera])
    return paths
