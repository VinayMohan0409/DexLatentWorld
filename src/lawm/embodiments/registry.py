"""The embodiment registry (configs/embodiments.json): which hands exist, how each robot cloud is sampled, and which the checkpoints saw."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

DEFAULT_REGISTRY = Path(__file__).resolve().parents[3] / "configs" / "embodiments.json"


@dataclass(frozen=True)
class Embodiment:
    """One registry entry."""

    name: str  # lowercase, as stored in the data
    kind: str  # "human" or "robot"
    trained: bool
    unidex_hand: Optional[str] = None
    exclude_links: Tuple[str, ...] = ()
    label: str = ""  # shown in plots and tables; empty = name

    @property
    def display(self) -> str:
        """Name used in plots and tables."""
        return self.label or self.name

    @property
    def seed_name(self) -> str:
        """Sampling-seed prefix: the UniDex folder name for unmodified stored hands (reproduces them), else the registry name."""
        if self.kind == "robot" and not self.exclude_links and self.name == (self.unidex_hand or "").lower():
            return self.unidex_hand
        return self.name


def load_registry(path: Optional[Path] = None) -> Dict[str, Embodiment]:
    """Read the registry in file order; checks kinds and that every robot names its UniDex hand."""
    raw = json.loads(Path(path or DEFAULT_REGISTRY).read_text())["embodiments"]
    out: Dict[str, Embodiment] = {}
    for name, spec in raw.items():
        if name != name.lower():
            raise ValueError(f"embodiment names are lowercase: {name}")
        unknown = set(spec) - {"kind", "trained", "unidex_hand", "exclude_links", "label", "_note"}
        if unknown:
            raise ValueError(f"{name}: unknown keys {sorted(unknown)}")
        e = Embodiment(name=name, kind=spec["kind"], trained=bool(spec["trained"]), unidex_hand=spec.get("unidex_hand"),
                       exclude_links=tuple(spec.get("exclude_links", ())), label=spec.get("label", ""))
        if e.kind not in ("human", "robot") or (e.kind == "robot") != bool(e.unidex_hand):
            raise ValueError(f"{name}: a robot needs unidex_hand, a human must not have one")
        out[name] = e
    return out
