---
name: vibescience-mcp-workflow
description: "Run the vibescience-mcp scientific-experiment-log loop end to end — pair deep research (perplexity_research) + arxiv MCP with a real, verdict-computed experiment cycle, then dogfood it into the markdown vault. Use when testing research hypotheses on real data and logging predicted-vs-observed causal claims for ML/analytics projects (e.g. optihealth)."
version: 1.0.0
author: Anton Ivanov / devpins.org
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [vibescience, mcp, research, experiment-log, perplexity, arxiv, optihealth, causal-map, calibration]
    related_skills: [ai-web-research, arxiv, server-admin]
---

# vibescience-mcp Workflow

`vibescience-mcp` (`/root/vibescience-mcp`, FastMCP stdio, vault `/root/vibescience-vault`) is a **scientific experiment log**: it stores *reasoning and causal claims*, not training curves, and defines **confirmed = predicted direction == observed direction** on a fixed diagnostic basis. Markdown files are truth; the SQLite index is disposable. Load this skill whenever you run a research→hypothesis→experiment→verdict loop and want it logged as a queryable causal map + calibration signal.

## The canonical loop (do it in this order)

1. **Deep research first** — `perplexity_research` (NOT `reason`/`ask`) for the 10-paper reading list. Pass FULL context: the exact stack, what already exists, and — critically — the **falsifiable-claim shape** you need back: *"intervention X moves metric Y in direction Z"* + which production diagnostic it moves + reproducibility rating. This makes every paper drop straight into a hypothesis.
2. **Verify arXiv ids** with the `arxiv` MCP (`get_abstract`) before trusting them — deep research sometimes hallucinates ids or mixes versions. Cheap insurance.
3. **Scope the tool class to the project.** optihealth_analysis is **classical statistics** (scipy/statsmodels/sklearn/ruptures/Prophet), NOT a learned encoder — that's optiHealth-**EiV**. Do NOT bring foundation/sensor-LM methods (WBM, LSM-2, PaPaGei, Chronos, MOMENT) into the analysis worker; they need GPUs/pretraining/raw sensors. If the first research pass returns the wrong class, re-run with an explicit **exclusion list** ("do NOT include X and nothing of the same class, do not repeat these ids").
4. **Register the fixed basis** — `register_diagnostic` for each production metric you'll predict on (anomaly_precision_at_k, anomaly_auroc, calibration_ece, cp_detection_delay, corr_fdr…), `register_intervention` for each named change, `add_paper` for grounding. **Tags are a registered basis too** — `list_tags(axis)` BEFORE tagging, and never coin a near-duplicate (see below).
5. **Propose the hypothesis with a committed prediction.** The FIRST `predicted_effect` is the *primary* one the verdict keys off. Pick the primary diagnostic that matches how the feature is actually consumed in prod (e.g. day_vector.py surfaces only top-5 anomalies → **precision@k**, not AUROC).
6. **Run the REAL experiment on REAL data**, then record before/after and let `close_experiment` **compute** the verdict. Never hand-assert "supports". Set `parent_experiment_id` when the run builds on a previous one. If the run **crashed**, call `abort_experiment(id, crash_reason)` — never invent a null result for it.
7. **Deploy only what the data earns** (see guardrail below).

## Tags are a REGISTERED vocabulary (2026-08 rework)

The tag layer was measured 69% dead before this was fixed: 18/36 tags used once,
20/36 confined to one entity kind, and `recall` only queried `hypotheses`, so a
tag carried only by papers returned `[]` and the agent concluded "no prior work".

- `list_tags(axis='topic'|'problem')` first. Reuse an existing tag.
- Different wording for an existing concept → add it as an **alias**, do NOT
  create a near-duplicate: `register_tag(id='<existing>', axis=..., aliases=['<yours>'])`.
- Unregistered tags are **rejected** with near-match suggestions — that error is
  the system working, not a bug to route around.
- Tags are inherited problem → hypothesis → experiment, so tag one entity well
  and the chain follows. `orphan: true` in `list_tags` = the tag connects nothing.
- Migrating an older vault: `python migrate_vault.py <vault> [--apply]`. It
  registers the existing vocabulary and *reports* merge candidates; it never
  auto-merges (collapsing `causality` into `causal-inference` would destroy a
  real distinction).

## The experiment DAG

`lineage(id)` / `children(id)` / `leaves(unevaluated_only=?)` answer what a flat
list cannot: the ancestry of a result, what was already tried on top of it, and
where the frontier is. Check `children()` before re-running an idea.

### When the experiment becomes a production statistical module

For irregular N-of-1 wearable data, do not jump directly from a paper shortlist to implementation. Follow the full pattern in [`references/n-of-1-training-analytics.md`](references/n-of-1-training-analytics.md): adversarially audit the research recommendation, measure *effective* post-overlap coverage, commit the estimand/gates before inspecting effects, verify exact one-to-one event attribution, recover known synthetic effects under TDD, and finally prove at the storage boundary that underpowered nominal findings were suppressed. A production null is a successful result; never relax gates because the real data self-silences.

## Hard-won pitfalls (this prompt cost real iterations)

- **perplexity_research hits a 180s client timeout by default.** Deep research legitimately runs 180–900s. If it times out, that is NOT a dead cookie (the "empty answer + valid backend_uuid" symptom is different — that's a stale token). The fix is raising the MCP call timeout; the `perplexity` config block carries its own `timeout:` (set to 900). Only the user can restart the gateway to apply it — ask, then retry the SAME call.
- **`perplexity_usage` / `/rest/rate-limit/all` returns 403 behind Cloudflare regardless of cookie validity.** Do not treat that 403 as "quota dead" — just call `perplexity_research` directly.
- **New MCP tools only appear in a NEW session.** After `hermes mcp add`, the tools are saved to config but not injected mid-session; the gateway must restart. `hermes mcp add` also prompts `Enable all N tools? [Y/n]` — with no TTY it cancels; pipe `printf 'Y\n' |` into it.
- **arxiv-mcp-server install trap:** install ONLY with `uv tool install 'arxiv-mcp-server[pdf]'`. `npm i arxiv-mcp-server` is an UNRELATED package; `uv pip install` doesn't put the binary on PATH. Add with `--storage-path /root/arxiv-papers`.
- **Make the experiment able to REFUTE.** First MCD-vs-Ledoit-Wolf run used 4σ injected outliers → both AUROC≈1.0 (trivially separable) → the claim's masking regime never activated → meaningless REFUTE. Fix: sweep the effect size (1.5–3σ) and use **correlated** (same-dims, same-sign) masking injection, the regime where a contaminated covariance actually breaks. A test that can't distinguish the hypotheses is worse than no test.
- **On unlabeled prod data, precision/recall needs a semi-synthetic protocol:** inject known contamination into the REAL background, then measure separability. It's the only honest way to get a number without ground-truth anomaly labels.
- **Log the FULL result, not the flattering slice.** If a method wins at eps=8% but regresses at eps=15%, record BOTH as separate experiments/diagnostics so `calibration` reflects reality (e.g. auroc accuracy 0.5, not a fake 1.0). Hiding the regression in `notes` while only measuring the win is subtle p-hacking. `recall` ranks refuted results first precisely to keep you honest.

## Deployment guardrail (guarded/conditional deploy)

A **regime-dependent** win is not a blanket win. If the data refutes the universal claim (MCD regresses AUROC at high contamination) but supports a narrow one (precision@k at moderate contamination), ship it as an **opt-in** (`robust_covariance` param, default off) with an **auto-fallback** on the failing condition (small n → Ledoit-Wolf), NOT as a default swap. Implement it yourself, supervised and verified — don't ban a valid change just because it isn't universal, and don't push a universal change the evidence doesn't support. Deploy exactly what the data earns, and not a pixel more.

> Terminology note: do NOT call this the "L2/L3 guardrail" — that conflates two
> unrelated things and Anton will correct it. "L3 Defender" in his profile is
> about *teaching* (steering his applied L2 impulses back to first-principles
> math), NOT about deployment scope. The deploy principle here is plain
> **"guarded / conditional deployment"**. The separate delegation rule ("a valid
> edit you'd make yourself is legitimate even if it's out of an unsupervised
> subagent's scope") is what justifies implementing the opt-in yourself.

## Dogfooding via the Store API

Drive the vault programmatically with the same layer the MCP tools call:
```python
import sys; sys.path.insert(0, "/root/vibescience-mcp/src")
from vibescience_mcp.core import Store
s = Store("/root/vibescience-vault")
s.register_diagnostic(...); s.register_intervention(...); s.add_paper(...)
s.create_problem(...); s.propose_hypothesis(..., predicted_effects=[{...}])
s.start_experiment(...); s.record_diagnostics(...)  # before/after
out = s.close_experiment(...)   # verdict is COMPUTED
s.calibration(diagnostic_id=...); s.causal_map(problem_id=...); s.recall(problem_id=...)
```
Deps in a fresh venv: `uv pip install python-frontmatter pydantic PyYAML` (plus pandas/sklearn/sqlalchemy/psycopg2-binary for the experiment). `uv pip`, never bare `pip` (venv isolation).

## Real prod data access (optihealth)

**Vault path:** the live vault is `/root/research/vibescience-vault` (the default
`/root/vibescience-vault` is an EMPTY decoy that already has an `.index.sqlite`,
so a misconfigured `VIBESCIENCE_VAULT` fails silently with `recall` returning
nothing). The repo is `/opt/vibescience-mcp` (`/root/vibescience-mcp` symlinks to
it — one repo, not two).

`DATABASE_URL` lives in `/opt/optihealth_analysis/src/.env.docker` (Supabase pooler, eu-north-1). Reuse the worker's query shape from `data_loader.py` / `database.py` (pivot `data_points` × `metric_definitions` → daily matrix). Build day-vectors exactly like `analysis/day_vector.py` (z-score, impute 0, +cyclical DoW) so the experiment matches production geometry. NOTE: the prod DB password was committed to git history and still needs rotation.

## Verification checklist

- [ ] `close_experiment` returned a COMPUTED verdict (not asserted).
- [ ] `recall(problem_id=...)` ranks the refuted hypothesis ABOVE the confirmed one, with a `why_it_failed` delta.
- [ ] `calibration` reflects BOTH wins and regressions (no fake 1.0).
- [ ] prod module still imports and runs in BOTH modes on real data before you call it deployed.
