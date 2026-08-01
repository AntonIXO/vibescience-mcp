"""The ``vibescience://guide`` resource text (brief §6, modelled on Basic
Memory's ai_assistant_guide). Kept as a module constant so it is importable in
tests and served verbatim by the MCP resource."""

GUIDE = """# vibescience-mcp — how to do science here (agent guide)

You are using a **scientific experiment log**, not a note dump. It stores
*reasoning and causal claims*, and defines **confirmed** as *agreement between a
prediction you committed BEFORE the test and the effect you observed AFTER*, on
a fixed set of diagnostics. Follow the order below. Skipping steps defeats the
whole point (a queryable causal map + a calibration signal on your own intuition).

## The loop (do it in this order)

1. **recall FIRST (pre-mortem gate).** Before proposing anything, call
   `recall(query/tags/problem_id)`. Refuted and inconclusive results are ranked
   to the TOP on purpose — they are the most valuable facts, because they stop
   you re-walking a dead end. Read `why_it_failed`. Read the `calibration_note`.
   `results` are hypotheses (they carry verdicts); `context` is related
   papers/interventions/problems — grounding, not evidence.
2. **Frame the Problem.** `create_problem` if the failure mode isn't logged yet.
3. **Register your basis.** Diagnostics are a *fixed, comparable* basis, not
   free text. `register_diagnostic` for any metric you'll predict on;
   `register_intervention` for any named change you'll apply.
   **Tags are a fixed basis too** — see below.
4. **Predict, THEN test.** `propose_hypothesis` REQUIRES ≥1 `predicted_effect`
   ({diagnostic_id, direction: up|down|none, magnitude_note}) on a registered
   diagnostic. The first predicted_effect is the *primary* one the verdict keys
   off. No prediction → rejected. This commitment is what makes calibration real.
5. **Run it.** `start_experiment(hypothesis_id)` — git_ref auto-captured from
   HEAD if you omit it. Set `parent_experiment_id` whenever this run builds on a
   previous one. `record_diagnostics` with before/after per diagnostic.
6. **Let the verdict be computed.** `close_experiment` computes
   `observed_effects`, `prediction_match` (per-diagnostic + overall), and the
   `verdict` (supports/refutes/inconclusive). You do NOT get to assert
   "confirmed" on vibes — it is derived from the primary prediction vs the
   observation. A positive match returns a *suggestion* to commit your git_ref;
   it never commits for you.
   **If the run CRASHED, call `abort_experiment(id, crash_reason)` instead.** A
   crash is not a result: it records no measurement, never enters calibration,
   and leaves the hypothesis retryable. Never invent a null result for it.

## Tags are a registered vocabulary, not free text

Free-text tagging rots. Measured on this vault before the registry existed:
18 of 36 tags were used exactly once, 20 of 36 lived on a single entity kind,
and a search for a tag carried only by papers returned *nothing* — so the agent
concluded "no prior work" while the evidence sat right there.

- `list_tags(axis)` BEFORE tagging. Reuse an existing tag.
- If your wording differs from an existing tag, do NOT coin a near-duplicate —
  add yours as an alias: `register_tag(id='<existing>', axis=..., aliases=['<yours>'])`.
  Aliases resolve on both write and query, and merging is additive.
- An unregistered tag is REJECTED with near-match suggestions.
- Two axes, never mixed: `topic` = subject matter, `problem` = failure mode.
- Tags are INHERITED (problem → hypothesis → experiment, intervention →
  hypothesis), so a tag is a real edge instead of a label on one lonely node.
- `orphan: true` in `list_tags` means the tag connects nothing. Fix it or drop it.

## The DAG: experiments have lineage

An experiment that builds on another must set `parent_experiment_id`. That one
link is what makes the log a search graph rather than a flat list:

- `children(id)` — what was already tried on top of this result. Check it before
  re-running an idea.
- `leaves(unevaluated_only=?)` — the frontier: work nothing was built on. Start
  here instead of branching from scratch.
- `lineage(id)` — the ancestry path, whose verdict chain *is* the research
  narrative.

## The payoff (query your own work)

- `causal_map(problem_id | tag)` → the aggregated `intervention → Δdiagnostic`
  subgraph across all experiments: "what has ever moved X, in which direction,
  how often, by how much." A mini meta-analysis of your own runs.
- `calibration(diagnostic_id | tag | intervention_id)` → where your intuition is
  miscalibrated (fraction of predictions that matched observation).

## Rules that are not negotiable

- Markdown files are the source of truth; the SQLite index is disposable
  (`reindex()` rebuilds it). Open the vault in Obsidian for graph/backlinks free.
- Negative results are first-class. Never delete a refuted hypothesis — supersede
  it (`propose_hypothesis(..., supersedes=<old_id>)`) so the dead-end chain stays.
- Reference external runs (W&B/MLflow) by id/url. This server does not store loss
  curves — it stores *why you thought it would work and whether it did*.
"""
