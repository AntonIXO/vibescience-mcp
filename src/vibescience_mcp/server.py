"""FastMCP server exposing vibescience over stdio (brief §6, §10).

Every tool description is written to *enforce the scientific ordering*
(recall → problem → register basis → predict → test → record → verdict) and
carries MCP behavior hints. The domain logic lives in ``core.Store``; this file
is a thin, well-annotated adapter.
"""

from __future__ import annotations

import os
from typing import Annotated, Any, Optional

from pydantic import Field

try:
    from fastmcp import FastMCP
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "fastmcp is required. Install with: pip install fastmcp"
    ) from e

from .core import Store, VibeScienceError
from .guide import GUIDE

VAULT = os.environ.get("VIBESCIENCE_VAULT", os.path.expanduser("~/vibescience-vault"))

mcp = FastMCP(
    name="vibescience-mcp",
    instructions=(
        "A scientific experiment log for ML research. Do science in order: "
        "recall past results (negative results ranked first) → frame a problem → "
        "register diagnostics/interventions → propose a hypothesis WITH a "
        "committed prediction → run an experiment → record diagnostics → close "
        "(verdict is computed, not asserted). Read the vibescience://guide "
        "resource before your first hypothesis."
    ),
)

_store: Optional[Store] = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store(VAULT)
    return _store


def _ok(data: Any) -> dict:
    return {"ok": True, "data": data}


def _err(e: Exception) -> dict:
    return {"ok": False, "error": str(e)}


def _dump(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [_dump(x) for x in obj]
    return obj


# --------------------------------------------------------------------------- #
# Guide resource
# --------------------------------------------------------------------------- #
@mcp.resource("vibescience://guide", mime_type="text/markdown")
def guide() -> str:
    """How to use this server in the correct scientific order. Read this first."""
    return GUIDE


# --------------------------------------------------------------------------- #
# recall — the pre-mortem gate
# --------------------------------------------------------------------------- #
@mcp.tool(
    annotations={"readOnlyHint": True, "idempotentHint": True, "destructiveHint": False}
)
def recall(
    query: Annotated[Optional[str], Field(description="Free-text about the idea you're about to try")] = None,
    topic_tags: Annotated[Optional[list[str]], Field(description="e.g. [jepa, contrastive]")] = None,
    problem_tags: Annotated[Optional[list[str]], Field(description="e.g. [attention-collapse]")] = None,
    problem_id: Optional[str] = None,
) -> dict:
    """CALL THIS FIRST, before proposing any hypothesis (pre-mortem gate).

    Returns ranked past hypotheses/experiments. **Refuted and inconclusive
    matches are ranked to the TOP** with a one-line reason they failed and the
    diagnostic delta that killed them — so you never silently re-walk a dead end.
    Also returns a calibration note if the query touches a diagnostic with a
    track record. After reading this, propose a NEW hypothesis (or supersede a
    prior one) with a committed prediction.
    """
    try:
        return _ok(store().recall(query, topic_tags, problem_tags, problem_id))
    except Exception as e:
        return _err(e)


# --------------------------------------------------------------------------- #
# Problems
# --------------------------------------------------------------------------- #
@mcp.tool(annotations={"idempotentHint": False, "destructiveHint": False})
def create_problem(
    title: str,
    description: str = "",
    topic_tags: Optional[list[str]] = None,
    problem_tags: Optional[list[str]] = None,
    paper_refs: Optional[list[str]] = None,
    id: Optional[str] = None,
) -> dict:
    """Register an open research question / failure mode. Do this once per
    distinct problem. Use `recall` first to check it isn't already logged."""
    try:
        return _ok(_dump(store().create_problem(
            title, description, topic_tags, problem_tags, paper_refs, id)))
    except Exception as e:
        return _err(e)


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def get_problem(id: str) -> dict:
    """Fetch a single problem by id."""
    try:
        return _ok(_dump(store().get_problem(id)))
    except Exception as e:
        return _err(e)


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def list_problems(status: Optional[str] = None, tags: Optional[list[str]] = None) -> dict:
    """List problems, optionally filtered by status {open|resolved|parked} or tags."""
    try:
        return _ok(_dump(store().list_problems(status, tags)))
    except Exception as e:
        return _err(e)


# --------------------------------------------------------------------------- #
# Diagnostics + interventions (the fixed basis)
# --------------------------------------------------------------------------- #
@mcp.tool(annotations={"idempotentHint": False, "destructiveHint": False})
def register_diagnostic(
    name: str,
    unit: str = "",
    direction: Annotated[str, Field(description="higher_better | lower_better | neutral")] = "neutral",
    description: str = "",
    id: Optional[str] = None,
) -> dict:
    """Register a named, fixed, measurable metric. Diagnostics are a DELIBERATE
    fixed basis — comparability across experiments is what makes the causal map
    possible. Do this before predicting on a metric."""
    try:
        return _ok(_dump(store().register_diagnostic(name, unit, direction, description, id)))
    except Exception as e:
        return _err(e)


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def list_diagnostics() -> dict:
    """List the registered diagnostic basis."""
    try:
        return _ok(_dump(store().list_diagnostics()))
    except Exception as e:
        return _err(e)


@mcp.tool(annotations={"idempotentHint": False, "destructiveHint": False})
def register_intervention(
    name: str, description: str = "", topic_tags: Optional[list[str]] = None, id: Optional[str] = None
) -> dict:
    """Register a named, reusable change (e.g. unfreeze-projector). Reused across
    hypotheses so the causal map can attribute deltas to it."""
    try:
        return _ok(_dump(store().register_intervention(name, description, topic_tags, id)))
    except Exception as e:
        return _err(e)


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def list_interventions() -> dict:
    """List registered interventions."""
    try:
        return _ok(_dump(store().list_interventions()))
    except Exception as e:
        return _err(e)


# --------------------------------------------------------------------------- #
# Hypothesis — prediction mandatory
# --------------------------------------------------------------------------- #
@mcp.tool(annotations={"idempotentHint": False, "destructiveHint": False})
def propose_hypothesis(
    problem_id: str,
    statement: str,
    predicted_effects: Annotated[
        list[dict],
        Field(description="REQUIRED ≥1: [{diagnostic_id, direction: up|down|none, magnitude_note}]. "
                          "The FIRST is the primary prediction the verdict keys off."),
    ],
    rationale: str = "",
    interventions: Optional[list[str]] = None,
    plan: str = "",
    topic_tags: Optional[list[str]] = None,
    problem_tags: Optional[list[str]] = None,
    papers: Optional[list[str]] = None,
    supersedes: Annotated[Optional[str], Field(description="id of a prior hypothesis this one revises")] = None,
    id: Optional[str] = None,
) -> dict:
    """Propose an explanation/intervention for a problem. You MUST commit ≥1
    `predicted_effect` on a REGISTERED diagnostic BEFORE testing — this is
    rejected otherwise. Committing the prediction up front is what turns this log
    into a calibration signal. Call `recall` first. To revise a dead end, pass
    `supersedes` rather than deleting it."""
    try:
        return _ok(_dump(store().propose_hypothesis(
            problem_id, statement, rationale, interventions, predicted_effects,
            plan, topic_tags, problem_tags, papers, supersedes, id)))
    except Exception as e:
        return _err(e)


# --------------------------------------------------------------------------- #
# Experiment lifecycle
# --------------------------------------------------------------------------- #
@mcp.tool(annotations={"idempotentHint": False, "destructiveHint": False})
def start_experiment(
    hypothesis_id: str,
    git_ref: Annotated[Optional[str], Field(description="branch@shortsha; auto-read from HEAD if omitted")] = None,
    external_run: Annotated[str, Field(description="W&B/MLflow run id or url")] = "",
    config_note: str = "",
    id: Optional[str] = None,
) -> dict:
    """Begin testing a hypothesis. If `git_ref` is omitted the server reads the
    current branch@commit from git HEAD. Reference the external W&B/MLflow run —
    this server stores verdicts, not loss curves."""
    try:
        return _ok(_dump(store().start_experiment(hypothesis_id, git_ref, external_run, config_note, id)))
    except Exception as e:
        return _err(e)


@mcp.tool(annotations={"idempotentHint": True, "destructiveHint": False})
def record_diagnostics(
    experiment_id: str,
    measurements: Annotated[list[dict], Field(description="[{diagnostic_id, before, after}]")],
) -> dict:
    """Record before/after values for measured diagnostics. Re-recording the same
    diagnostic overwrites it (idempotent). Observed deltas/directions are derived."""
    try:
        return _ok(_dump(store().record_diagnostics(experiment_id, measurements)))
    except Exception as e:
        return _err(e)


@mcp.tool(annotations={"idempotentHint": True, "destructiveHint": False})
def close_experiment(experiment_id: str, notes: str = "") -> dict:
    """Finalize: compute observed_effects, prediction_match (per-diagnostic +
    overall) and the verdict (supports/refutes/inconclusive), then propagate the
    status to the hypothesis. The verdict is COMPUTED from the primary prediction
    vs the observation — you cannot assert 'confirmed'. On a positive match it
    RETURNS a suggestion to commit your git_ref; it never commits for you."""
    try:
        return _ok(store().close_experiment(experiment_id, notes))
    except Exception as e:
        return _err(e)


# --------------------------------------------------------------------------- #
# Papers
# --------------------------------------------------------------------------- #
@mcp.tool(annotations={"idempotentHint": False, "destructiveHint": False})
def add_paper(
    title: str,
    arxiv_id_or_url: str = "",
    key_claims: Optional[list[str]] = None,
    topic_tags: Optional[list[str]] = None,
    id: Optional[str] = None,
) -> dict:
    """Add an external reference (metadata + key claims). Cite it from hypotheses
    to ground your rationale."""
    try:
        return _ok(_dump(store().add_paper(title, arxiv_id_or_url, key_claims, topic_tags, id)))
    except Exception as e:
        return _err(e)


@mcp.tool(annotations={"idempotentHint": True, "destructiveHint": False})
def link_paper(entity_id: str, paper_id: str) -> dict:
    """Link a paper to a problem or hypothesis (adds a cites edge)."""
    try:
        return _ok(store().link_paper(entity_id, paper_id))
    except Exception as e:
        return _err(e)


# --------------------------------------------------------------------------- #
# The payoff: causal map + calibration
# --------------------------------------------------------------------------- #
@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def causal_map(
    problem_id: Optional[str] = None,
    tag: Optional[str] = None,
    emit_canvas: Annotated[bool, Field(description="also write an Obsidian .canvas")] = False,
) -> dict:
    """Aggregated intervention → Δdiagnostic subgraph across all experiments in
    scope: what has ever moved a diagnostic, in which direction, how often, by how
    much. A mini meta-analysis of your own runs. Scope by problem_id or tag."""
    try:
        s = store()
        res = s.causal_map(problem_id, tag)
        if emit_canvas:
            res = dict(res)
            res["canvas_path"] = s.write_canvas(problem_id, tag)
        return _ok(res)
    except Exception as e:
        return _err(e)


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def calibration(
    diagnostic_id: Optional[str] = None,
    tag: Optional[str] = None,
    intervention_id: Optional[str] = None,
) -> dict:
    """Prediction-accuracy report: the fraction of experiments where predicted
    direction == observed direction. Answers 'where is my intuition
    miscalibrated?'. Scope by diagnostic, tag, or intervention."""
    try:
        return _ok(store().calibration(diagnostic_id, tag, intervention_id))
    except Exception as e:
        return _err(e)


@mcp.tool(annotations={"idempotentHint": True, "destructiveHint": False})
def reindex() -> dict:
    """Rebuild the disposable SQLite index from the markdown vault (files are
    truth). Safe to call anytime; produces identical query results."""
    try:
        return _ok(store().reindex())
    except Exception as e:
        return _err(e)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
