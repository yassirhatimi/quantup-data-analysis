"""Data-quality accounting.

Every fetch in this system reports back through here, so the dashboard can
state exactly what the prediction was and was not built from:

    Market price:   live
    Polymarket:     unavailable (no relevant market)
    News sentiment: delayed (18 min)

The rule the rest of the engine obeys: a source that failed contributes
*nothing* and its model weight is redistributed, rather than a zero score being
quietly blended in as if it were evidence of balance. Missing data must move
confidence down, never nudge the direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

#: Ordered worst-to-best so `min()` over a set of statuses is meaningful.
#: "closed" is its own state: the price is real and final, it is simply not
#: moving, which is a different thing from a feed that has gone quiet.
STATUS_RANK = {"unavailable": 0, "stale": 1, "closed": 2, "delayed": 2,
               "live": 3, "n/a": 3}


@dataclass
class SourceStatus:
    """The state of one data source for one run."""
    name: str
    status: str                  # live | delayed | closed | stale | unavailable | n/a
    detail: str = ""
    as_of: pd.Timestamp | None = None
    age_seconds: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("live", "delayed", "closed", "n/a")

    def describe(self) -> str:
        bits = [self.status]
        if self.age_seconds is not None and self.status in ("delayed", "stale"):
            bits.append(_age(self.age_seconds))
        if self.detail:
            bits.append(self.detail)
        if self.error:
            bits.append(self.error)
        return f"{self.name}: " + " · ".join(bits)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "status": self.status, "detail": self.detail,
            "as_of": str(self.as_of) if self.as_of is not None else None,
            "age_seconds": self.age_seconds, "error": self.error,
        }


def _age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s old"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min old"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h old"
    return f"{seconds / 86400:.1f}d old"


def age_status(age_seconds: float | None, live_s: float, stale_s: float) -> str:
    """Classify a feed by how old its newest observation is."""
    if age_seconds is None:
        return "unavailable"
    if age_seconds <= live_s:
        return "live"
    if age_seconds <= stale_s:
        return "delayed"
    return "stale"


@dataclass
class QualityReport:
    """All source statuses for one prediction, plus the headline verdict."""
    sources: list[SourceStatus] = field(default_factory=list)
    availability: float = 0.0     # share of model weight backed by real data
    label: str = "LIMITED"        # GOOD | LIMITED | POOR
    reasons: list[str] = field(default_factory=list)

    def add(self, status: SourceStatus) -> SourceStatus:
        self.sources.append(status)
        return status

    def get(self, name: str) -> SourceStatus | None:
        for s in self.sources:
            if s.name == name:
                return s
        return None

    def unavailable(self) -> list[SourceStatus]:
        return [s for s in self.sources if s.status == "unavailable"]

    def stale(self) -> list[SourceStatus]:
        return [s for s in self.sources if s.status == "stale"]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "availability": round(self.availability, 3),
            "reasons": self.reasons,
            "sources": [s.to_dict() for s in self.sources],
        }


def grade(report: QualityReport, availability: float, price_status: str,
          good: float, limited: float) -> QualityReport:
    """Set the headline GOOD / LIMITED / POOR verdict and say why.

    Price is privileged: with no usable price there is no prediction worth
    printing, whatever else arrived.
    """
    report.availability = availability
    reasons: list[str] = []

    if price_status == "unavailable":
        report.label = "POOR"
        report.reasons = ["no usable market price this run"]
        return report

    if price_status == "stale":
        reasons.append("market price is stale")
    elif price_status == "delayed":
        reasons.append("market price is delayed")
    elif price_status == "closed":
        reasons.append("market is closed — price is the last traded close")

    missing = [s.name for s in report.unavailable()]
    if missing:
        reasons.append("unavailable: " + ", ".join(missing))
    stale = [s.name for s in report.stale()]
    if stale:
        reasons.append("stale: " + ", ".join(stale))

    if availability >= good and price_status == "live" and not stale:
        label = "GOOD"
    elif availability >= limited and price_status != "stale":
        label = "LIMITED"
    elif availability >= limited:
        label = "LIMITED"
    else:
        label = "POOR"

    report.label = label
    report.reasons = reasons or ["all tracked sources responded"]
    return report
