from __future__ import annotations

from pathlib import Path


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def iter_images(input_path: str | Path, extensions: tuple[str, ...] = DEFAULT_EXTENSIONS):
    path = Path(input_path)
    if path.is_file():
        yield path
        return

    normalized = tuple(ext.lower() for ext in extensions)
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in normalized:
            yield candidate
