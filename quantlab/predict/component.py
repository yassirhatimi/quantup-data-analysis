"""The common currency of the engine: one evidence channel's read.

Every component — technical, macro, Polymarket, whatever gets added later —
returns a `ComponentScore`. That uniformity is what lets the engine reweight,
renormalise and explain them generically, and what makes "this source was
missing" a first-class outcome rather than a silent zero.

Score convention, everywhere, no exceptions:

    +1  maximally bullish for this asset
     0  no directional information
    -1  maximally bearish

`available=False` means the channel had no usable data. Its weight is
redistributed across the channels that did, and its absence is charged against
model confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ComponentScore:
    name: str
    score: float = 0.0
    available: bool = True
    weight_hint: float = 1.0          # within-component strength, 0..1
    evidence: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    status: str = "live"
    note: str = ""

    def __post_init__(self) -> None:
        self.score = clamp(float(self.score))
        if not self.available:
            self.score = 0.0

    @property
    def direction(self) -> str:
        if self.score > 0.05:
            return "bullish"
        if self.score < -0.05:
            return "bearish"
        return "neutral"

    def to_dict(self) -> dict:
        return {
            "name": self.name, "score": round(self.score, 4),
            "available": self.available, "status": self.status,
            "evidence": self.evidence, "detail": _jsonable(self.detail),
            "note": self.note,
        }


def missing(name: str, reason: str, status: str = "unavailable") -> ComponentScore:
    """A channel that had nothing to say, and says so."""
    return ComponentScore(name=name, score=0.0, available=False,
                          status=status, note=reason)


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(np.clip(x, lo, hi))


def squash(x: float, scale: float) -> float:
    """Map an unbounded reading onto [-1, 1]; `scale` is the ~0.76 point.

    tanh rather than a hard clip so that a driver three standard deviations out
    is meaningfully stronger than one at two, without ever running away.
    """
    if x is None or not np.isfinite(x) or scale <= 0:
        return 0.0
    return float(np.tanh(x / scale))


def blend(parts: list[tuple[float, float]]) -> float:
    """Weighted mean of (value, weight) pairs, ignoring zero-weight entries."""
    usable = [(v, w) for v, w in parts if w > 0 and v is not None and np.isfinite(v)]
    if not usable:
        return 0.0
    total = sum(w for _, w in usable)
    return clamp(sum(v * w for v, w in usable) / total)


def _jsonable(obj):
    """numpy / pandas scalars are not JSON-serialisable; make them so."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
