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
2. **Frame the Problem.** `create_problem` if the failure mode isn't logged yet.
3. **Register your basis.** Diagnostics are a *fixed, comparable* basis, not
   free text. `register_diagnostic` for any metric you'll predict on;
   `register_intervention` for any named change you'll apply.
4. **Predict, THEN test.** `propose_hypothesis` REQUIRES ≥1 `predicted_effect`
   ({diagnostic_id, direction: up|down|none, magnitude_note}) on a registered
   diagnostic. The first predicted_effect is the *primary* one the verdict keys
   off. No prediction → rejected. This commitment is what makes calibration real.
5. **Run it.** `start_experiment(hypothesis_id)` — git_ref auto-captured from
   HEAD if you omit it. `record_diagnostics` with before/after per diagnostic.
6. **Let the verdict be computed.** `close_experiment` computes
   `observed_effects`, `prediction_match` (per-diagnostic + overall), and the
   `verdict` (supports/refutes/inconclusive). You do NOT get to assert
   "confirmed" on vibes — it is derived from the primary prediction vs the
   observation. A positive match returns a *suggestion* to commit your git_ref;
   it never commits for you.

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
