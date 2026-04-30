from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class OntologyItem:
    prompt: str
    class_name: str
    class_id: int


def load_ontology(path: str | Path) -> list[OntologyItem]:
    """Load a prompt-to-class ontology from YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Ontology must be a YAML mapping of prompt: class_name")

    class_ids: dict[str, int] = {}
    items: list[OntologyItem] = []

    for prompt, class_name in raw.items():
        prompt_text = str(prompt).strip()
        class_text = str(class_name).strip()
        if not prompt_text or not class_text:
            continue
        if class_text not in class_ids:
            class_ids[class_text] = len(class_ids)
        items.append(
            OntologyItem(
                prompt=prompt_text,
                class_name=class_text,
                class_id=class_ids[class_text],
            )
        )

    if not items:
        raise ValueError(f"No ontology entries found in {path}")

    return items


def class_names(items: list[OntologyItem]) -> list[str]:
    names_by_id = {item.class_id: item.class_name for item in items}
    return [names_by_id[index] for index in sorted(names_by_id)]
