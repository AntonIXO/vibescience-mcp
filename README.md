# vibescience-mcp

A **scientific experiment log** (MCP server) for an AI coding/research agent working on ML projects.

It is *not* a lab notebook and *not* Weights & Biases. It stores **reasoning and causal claims**, not training curves, and defines **"confirmed"** as *agreement between a prediction committed **before** the test and the effect observed **after***, on a **fixed diagnostic basis**. That makes your log a queryable **causal map** ("which interventions move which diagnostics, in which direction, how often") plus a **calibration signal** on your own intuition.

Markdown files are the source of truth → open `vault/` in **Obsidian** for graph view, backlinks and tag panels for free. The SQLite index is disposable and rebuilt from markdown at any time.

## Why it's different

| | lab notebook | W&B / MLflow | **vibescience** |
|---|---|---|---|
| stores | free-text notes | loss curves, metrics | **predictions, verdicts, causal claims** |
| "confirmed" means | you say so | — | **predicted direction == observed direction** (computed) |
| negative results | buried | buried | **first-class, ranked to the top of recall** |
| cross-experiment view | none | per-run dashboards | **aggregated causal map + calibration** |

## Install

```bash
cd vibescience-mcp
python -m venv .venv && . .venv/bin/activate
pip install -e .            # add ".[embeddings]" for optional semantic recall, ".[dev]" for tests
```

## Configure in Claude Code / Cursor / Desktop

The server speaks **stdio**. Point your MCP client at it and set the vault path:

```json
{
  "mcpServers": {
    "vibescience": {
      "command": "/root/vibescience-mcp/.venv/bin/python",
      "args": ["-m", "vibescience_mcp.server"],
      "env": {
        "VIBESCIENCE_VAULT": "/root/vibescience-vault",
        "VIBESCIENCE_EMBEDDINGS": "off"
      }
    }
  }
}
```

Or via the Claude Code CLI:

```bash
claude mcp add vibescience -e VIBESCIENCE_VAULT=/root/vibescience-vault \
  -- /root/vibescience-mcp/.venv/bin/python -m vibescience_mcp.server
```

## The scientific loop (enforced by tool descriptions)

1. **`recall`** — *always first* (pre-mortem gate). Refuted/inconclusive matches rank to the **top**, each with a one-line failure reason and the diagnostic delta that killed it. Never re-walk a dead end.
2. **`create_problem`** — frame the open question / failure mode.
3. **`register_diagnostic`** / **`register_intervention`** — diagnostics are a *fixed, comparable basis*, not free text. Adding one is deliberate.
4. **`propose_hypothesis`** — **requires ≥1 `predicted_effect`** on a registered diagnostic *before* testing. The first is the *primary* prediction the verdict keys off. No prediction → rejected.
5. **`start_experiment`** — auto-captures `branch@shortsha` from git HEAD if you omit `git_ref`. References an external W&B/MLflow run; never stores curves.
6. **`record_diagnostics`** — before/after per diagnostic; deltas/directions derived.
7. **`close_experiment`** — **computes** `observed_effects`, `prediction_match` (per-diagnostic + overall) and the `verdict` (`supports`/`refutes`/`inconclusive`), propagates status to the hypothesis, and **suggests** (never performs) a commit on a positive match.

Then query your own work:

- **`causal_map(problem_id | tag)`** — aggregated `intervention → Δdiagnostic` subgraph. Optional Obsidian `.canvas` output.
- **`calibration(diagnostic_id | tag | intervention_id)`** — fraction of predictions that matched observation. Where is your intuition wrong?
- **`reindex()`** — rebuild the SQLite index from markdown (idempotent).

A **`vibescience://guide`** resource ships the same workflow to any agent.

## Storage layout

```
vault/
  problems/        hypotheses/     experiments/
  diagnostics/     interventions/  papers/
  _canvas/         # generated Obsidian canvases
  .index.sqlite    # disposable — rebuilt from markdown by reindex()
```

Each entity is one markdown file: YAML frontmatter for structured fields, body for prose + `[[wikilinks]]`. Fully Obsidian-compatible.

## Env vars

| var | default | meaning |
|---|---|---|
| `VIBESCIENCE_VAULT` | `~/vibescience-vault` | vault path (source of truth) |
| `VIBESCIENCE_EMBEDDINGS` | `off` | `on` enables local FastEmbed semantic recall (Phase 2 scaffold) |

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # 24 tests: verdict math, prediction gate, full loop,
                   # negative-result ranking, causal_map, calibration,
                   # index rebuild, Obsidian compat, MCP stdio smoke test
```

## Phasing

- **Phase 1 (this):** schema + lifecycle + `recall` + `causal_map` + `calibration` over markdown, SQLite index, Obsidian compatibility.
- **Phase 2 (scaffolded, not built):** hold papers in context — fetch arXiv full text, chunk, embed, and let `recall` pull relevant passages in. The `Paper` schema is ready.

**Non-goals:** custom graph visualization (Obsidian does it), web UI, autonomous experiment execution, raw metric-curve storage (reference W&B/MLflow), auto-committing to git.

## License

MIT
