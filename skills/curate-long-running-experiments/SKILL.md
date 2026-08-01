---
name: curate-long-running-experiments
description: Delegate long-running, sequential ML experiments to one economical high-reasoning subagent while the primary agent retains scientific decisions and independently verifies frozen inputs, completed artifacts, gates, and verdicts. Use for multi-seed GPU campaigns, benchmark matrices, confirmation runs, locked evaluations, or experiment babysitting where repeated polling consumes primary-agent context; especially when the user requests subagents, limited context, smaller models, no parallel GPU use, interruption recovery, or factual artifact-backed results.
---

# Curate Long-Running Experiments

Keep scientific authority with the primary agent. Give one economical curator
exclusive ownership of mechanical execution, waiting, milestone collection,
and first-pass artifact authentication.

Read [references/curator-brief.md](references/curator-brief.md) when composing
the curator task.

## 1. Separate authority from execution

The primary agent must:

1. Recall prior results and register the hypothesis, diagnostics, prediction,
   interventions, and gates before scientific runs.
2. Inspect the repository and freeze exact commands, source/config/data/cache
   hashes, seeds, order, precision, hardware, and test-lock policy.
3. Decide whether interrupted or failed runs are scientifically admissible.
4. Independently verify final artifacts and compute or audit the conclusion.
5. Close the scientific record and make commits, deployments, or test-unlock
   decisions.

Delegate only:

- running already-frozen commands;
- ensuring one accelerator process at a time;
- waiting on the active process internally;
- authenticating report sidecars and artifact manifests;
- reporting sparse milestones;
- running a preregistered aggregation command;
- stopping at a frozen gate or anomaly.

Do not delegate hypothesis revision, hyperparameter changes, result selection,
source edits, destructive cleanup, deployment, or interpretation of ambiguous
evidence.

## 2. Choose an economical curator

Inspect the model overrides advertised by `spawn_agent`. Prefer a smaller,
lower-cost agent with high reasoning because curation is procedural but
integrity-sensitive.

- Use the exact economical model requested by the user when it is available.
- Otherwise choose the least costly advertised capable model. In environments
  advertising `gpt-5.6-terra`, prefer it with `reasoning_effort: "xhigh"`.
- Never invent or claim availability of a model such as Luna when the tool does
  not advertise it.
- Use `fork_turns: "none"` when setting a model override. Pass a compact frozen
  brief instead of leaking the full conversation.
- Omit the override rather than failing if no economy model is advertised.

Spawn exactly one curator for a shared GPU campaign. Do not create one agent per
seed unless runs use isolated resources and the user explicitly permits
parallelism.

## 3. Freeze before spawning

Do not let the curator infer the protocol from prose. Give it:

- canonical workspace and output paths;
- immutable git/worktree, implementation, protocol, config, data, and cache
  hashes;
- exact encoder checkpoint/revision/tokenizer identity, not only a model family;
- exact commands in exact order;
- expected seeds, steps, samples, and hardware;
- allowed and forbidden actions;
- retry and interruption policy;
- required report fields and checksum procedure;
- aggregation command and gate behavior;
- a clear stop boundary before locked test, deployment, or mutation.

If any required choice is not frozen, resolve it as the primary agent before
delegation.

## 4. Make the curator own waiting

The curator must launch one run and wait on its process/session until completion
before launching the next. It must not ask the primary agent to poll the GPU,
metrics log, filesystem, or process table.

Require messages only when:

- one paired seed (or other natural comparison unit) finishes and authenticates;
- an anomaly, drift, OOM, nonfinite value, or interruption occurs;
- an aggregate finishes;
- the whole bounded task finishes.

While the curator works, the primary agent should use the agent wait mechanism,
not repeated shell diagnostics. Send concise no-change user updates only as
required by the product's communication cadence. Do not duplicate the
curator's monitoring.

## 5. Enforce a result-neutral run policy

The curator must:

- run seeds and arms in preregistered order;
- retain every completed result;
- never retry based on loss or metrics;
- never substitute a seed;
- never open a locked test early;
- never alter source/config/cache inputs;
- never run experiments concurrently on a shared accelerator;
- release and verify the accelerator between units;
- stop on integrity drift instead of repairing the protocol.

A smoke run is engineering evidence only unless explicitly preregistered as
scientific evidence.

## 6. Recover from interruptions without selection

After a parent or subagent interruption:

1. Inspect read-only process state and expected report status.
2. If a valid process is still running, do not launch a duplicate.
3. If a report is complete, authenticate it and continue without rerunning.
4. If no process exists and the report is incomplete, determine the last logged
   step and whether any final evaluation/outcome was produced.
5. Preserve the incomplete directory verbatim under an explicit
   `interrupted_runs/` path; never delete or overwrite it.
6. Restart the exact same seed only when the interruption was external and no
   final outcome was observed. Record that recovery in the final report.
7. If an outcome was observed or admissibility is ambiguous, stop for the
   primary agent's decision.

Do not call an outcome-unobserved restart a scientific retry.

## 7. Verify independently after handoff

Treat curator summaries as indexes, not proof. The primary agent must read the
actual final artifacts and verify:

- frozen implementation, protocol, config, data, split, and cache hashes;
- required model/encoder identity and fallback prohibition;
- complete seed × arm matrix with exact order-independent membership;
- status, steps, samples, precision, GPU, OOM, and nonfinite fields;
- exact model revision/tokenizer identity and baseline worktree-state match;
- identical paired initialization, evaluation cohort, caption hashes, and
  exposure where required;
- report sidecar hashes and every artifact-manifest entry;
- paired metric deltas, exact tests, bootstrap settings, and every gate;
- absence or validity of test/deployment unlocks;
- preservation of interrupted artifacts and no unexpected source changes.

Recompute aggregates from reports when practical. Investigate any mismatch
before accepting a verdict.

## 8. Finish the scientific loop

Record factual before/after diagnostics only after verification. Let the
science system compute the verdict. Distinguish:

- directional support;
- preregistered magnitude-gate pass;
- deployment/collapse-gate pass;
- locked-test eligibility.

Do not equate one with another. Commit only verified source and intended
artifacts, preserving unrelated user changes.

## Example spawn

Use only if the named model appears in the current `spawn_agent` schema:

```text
spawn_agent(
  task_name="curate_confirmation",
  fork_turns="none",
  model="gpt-5.6-terra",
  reasoning_effort="xhigh",
  message=<completed frozen curator brief>,
)
```

The primary agent then waits for milestone messages and performs the independent
post-run audit.
