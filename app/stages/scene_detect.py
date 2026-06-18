"""Scene detection — extract one keyframe per scene.

Uses PySceneDetect's ContentDetector for scene boundary detection.
Threshold ~27 is the library default for cuts; lower (e.g. 15) for more
sensitive cuts in CCTV-style content.
"""
import subprocess
from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video


def _scene_list(video_path: Path, threshold: float) -> list[tuple[float, float]]:
    """Return [(start_sec, end_sec), ...]."""
    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()
    if not scenes:
        # Whole video is one scene
        return [(0.0, video.duration.get_seconds())]
    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]


def _grab_frame(video_path: Path, t_sec: float, out_path: Path) -> bool:
    """Extract a single frame at t_sec via ffmpeg."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{t_sec:.3f}", "-i", str(video_path),
        "-frames:v", "1", "-q:v", "3", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and out_path.exists()


def extract_scene_keyframes(video_path: Path, out_dir: Path,
                            threshold: float = 27.0,
                            max_per_scene: int = 1) -> list[Path]:
    """For each scene, extract `max_per_scene` evenly spaced keyframes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = _scene_list(video_path, threshold)
    saved = []
    stem = video_path.stem
    for idx, (start, end) in enumerate(scenes):
        if max_per_scene == 1:
            pts = [(start + end) / 2.0]
        else:
            step = (end - start) / (max_per_scene + 1)
            pts = [start + step * (k + 1) for k in range(max_per_scene)]
        for j, t in enumerate(pts):
            out = out_dir / f"{stem}_s{idx:04d}_{j}.jpg"
            if _grab_frame(video_path, t, out):
                saved.append(out)
    return saved
