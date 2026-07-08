"""Pydantic schemas + the verdict / prediction-match math.

Markdown files are the source of truth (see brief §2.6). These models exist to
validate structured frontmatter and to *compute* verdicts deterministically —
the agent may not hand-wave a ``confirmed`` (brief §5).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

EPS = 1e-9  # below this |delta| a diagnostic is considered "did not move"


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Direction(str, Enum):
    """Direction of a predicted or observed effect on a diagnostic."""

    up = "up"
    down = "down"
    none = "none"


class DiagDirection(str, Enum):
    """Whether higher/lower is better for a diagnostic (registry metadata)."""

    higher_better = "higher_better"
    lower_better = "lower_better"
    neutral = "neutral"


class ProblemStatus(str, Enum):
    open = "open"
    resolved = "resolved"
    parked = "parked"


class HypothesisStatus(str, Enum):
    proposed = "proposed"
    testing = "testing"
    confirmed = "confirmed"
    refuted = "refuted"
    inconclusive = "inconclusive"


class Verdict(str, Enum):
    supports = "supports"
    refutes = "refutes"
    inconclusive = "inconclusive"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_slug_re = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Deterministic slug for entity IDs / filenames."""
    s = _slug_re.sub("-", text.strip().lower()).strip("-")
    return s or "untitled"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def delta_direction(before: float, after: float, eps: float = EPS) -> Direction:
    d = after - before
    if abs(d) <= eps:
        return Direction.none
    return Direction.up if d > 0 else Direction.down


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
class PredictedEffect(BaseModel):
    diagnostic_id: str
    direction: Direction
    magnitude_note: str = ""


class ObservedEffect(BaseModel):
    diagnostic_id: str
    delta: float
    direction: Direction


class Measurement(BaseModel):
    diagnostic_id: str
    before: float
    after: float


class PredictionMatch(BaseModel):
    """Per-diagnostic match booleans plus an overall verdict flag."""

    per_diagnostic: dict[str, bool] = Field(default_factory=dict)
    overall: bool = False


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #
class Diagnostic(BaseModel):
    id: str
    name: str
    unit: str = ""
    direction: DiagDirection = DiagDirection.neutral
    description: str = ""


class Intervention(BaseModel):
    id: str
    name: str
    description: str = ""
    topic_tags: list[str] = Field(default_factory=list)


class Paper(BaseModel):
    id: str
    title: str
    arxiv_id_or_url: str = ""
    key_claims: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)


class Problem(BaseModel):
    id: str
    title: str
    description: str = ""
    status: ProblemStatus = ProblemStatus.open
    topic_tags: list[str] = Field(default_factory=list)
    problem_tags: list[str] = Field(default_factory=list)
    paper_refs: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


class Hypothesis(BaseModel):
    id: str
    problem_id: str
    statement: str
    rationale: str = ""
    interventions: list[str] = Field(default_factory=list)
    predicted_effects: list[PredictedEffect] = Field(default_factory=list)
    plan: str = ""
    status: HypothesisStatus = HypothesisStatus.proposed
    topic_tags: list[str] = Field(default_factory=list)
    problem_tags: list[str] = Field(default_factory=list)
    paper_refs: list[str] = Field(default_factory=list)
    supersedes: Optional[str] = None  # id of a hypothesis this one revises
    created_at: str = Field(default_factory=now_iso)

    @field_validator("predicted_effects")
    @classmethod
    def _require_prediction(cls, v: list[PredictedEffect]) -> list[PredictedEffect]:
        if not v:
            raise ValueError(
                "A hypothesis requires >=1 predicted_effect referencing a "
                "registered diagnostic (brief §6). Commit a prediction before testing."
            )
        return v

    @property
    def primary_prediction(self) -> PredictedEffect:
        return self.predicted_effects[0]


class Experiment(BaseModel):
    id: str
    hypothesis_id: str
    git_ref: str = ""
    external_run: str = ""
    config_note: str = ""
    diagnostics_measured: list[Measurement] = Field(default_factory=list)
    observed_effects: list[ObservedEffect] = Field(default_factory=list)
    verdict: Optional[Verdict] = None
    prediction_match: Optional[PredictionMatch] = None
    notes: str = ""
    artifacts: list[str] = Field(default_factory=list)
    closed: bool = False
    created_at: str = Field(default_factory=now_iso)


# --------------------------------------------------------------------------- #
# The core computation: predicted-vs-observed (brief §5)
# --------------------------------------------------------------------------- #
def compute_observed_effects(
    measurements: list[Measurement], eps: float = EPS
) -> list[ObservedEffect]:
    out: list[ObservedEffect] = []
    for m in measurements:
        out.append(
            ObservedEffect(
                diagnostic_id=m.diagnostic_id,
                delta=round(m.after - m.before, 12),
                direction=delta_direction(m.before, m.after, eps),
            )
        )
    return out


def compute_prediction_match(
    predicted: list[PredictedEffect],
    observed: list[ObservedEffect],
) -> PredictionMatch:
    """Per-diagnostic direction agreement + overall (primary-prediction) flag.

    ``overall`` is driven by the *primary* predicted effect (the first one) —
    this is what the verdict keys off of. A diagnostic that was predicted but
    never measured counts as a non-match.
    """
    obs_by_id = {o.diagnostic_id: o for o in observed}
    per: dict[str, bool] = {}
    for p in predicted:
        o = obs_by_id.get(p.diagnostic_id)
        per[p.diagnostic_id] = bool(o is not None and o.direction == p.direction)

    overall = False
    if predicted:
        primary = predicted[0]
        overall = per.get(primary.diagnostic_id, False)
    return PredictionMatch(per_diagnostic=per, overall=overall)


def compute_verdict(
    predicted: list[PredictedEffect],
    observed: list[ObservedEffect],
) -> Verdict:
    """Compute the experiment verdict from the *primary* predicted effect.

    - ``supports``      primary observed direction == primary predicted direction
    - ``refutes``       primary observed direction is the strict opposite
    - ``inconclusive``  primary diagnostic did not move, was not measured,
                        or the prediction was a null (``none``) that failed
    """
    if not predicted:
        return Verdict.inconclusive
    primary = predicted[0]
    obs_by_id = {o.diagnostic_id: o for o in observed}
    o = obs_by_id.get(primary.diagnostic_id)
    if o is None or o.direction == Direction.none:
        return Verdict.inconclusive
    if o.direction == primary.direction:
        return Verdict.supports
    # observed moved, and in a named direction different from prediction
    opposite = {Direction.up: Direction.down, Direction.down: Direction.up}
    if primary.direction in opposite and o.direction == opposite[primary.direction]:
        return Verdict.refutes
    # predicted 'none' but it moved → the null prediction failed → refutes
    return Verdict.refutes


def verdict_to_hypothesis_status(v: Verdict) -> HypothesisStatus:
    return {
        Verdict.supports: HypothesisStatus.confirmed,
        Verdict.refutes: HypothesisStatus.refuted,
        Verdict.inconclusive: HypothesisStatus.inconclusive,
    }[v]
