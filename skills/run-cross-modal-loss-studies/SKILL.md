---
name: run-cross-modal-loss-studies
description: Design, implement, and execute reproducible cross-modal representation-loss studies on real data. Use when comparing an incumbent contrastive objective with kernel/HSIC/CKA/KTA, CCA, VICReg, Barlow-style, or multi-positive alternatives; building deterministic GPU pilot and confirmation campaigns; auditing duplicate positives, split leakage, probe saturation, collapse, or learned loss-head bypass; or deciding whether pilot evidence earns a full test.
---

# Run Cross-Modal Loss Studies

Use this workflow to turn an objective idea into a falsifiable, resource-safe
experiment. Pair it with `$vibescience-mcp-workflow` when Vibescience tools are
available.

Read [references/eiv-case-study.md](references/eiv-case-study.md) when working
on sensor-text representations, OptiHealth EiV, or a campaign showing high
probe scores but poor embedding geometry.

## Preserve the order

### 1. Define three distinct baselines

Measure and preserve:

1. **Seed-matched untrained control**: the exact initialized model before any
   optimizer step.
2. **Trained incumbent**: the current objective under the same seed, data,
   compute, and evaluation.
3. **Alternative**: change only the registered objective/intervention.

Never call the untrained snapshot the previous loss. Never compare an
alternative only with random initialization when the question is whether it
beats the incumbent. Use paired seeds for confirmation.

### 2. Audit the data path before training

Verify:

- subject/entity-disjoint train, validation, and locked-test splits;
- normalization statistics are fitted on training data only;
- missing sentinels are not treated as physiological values;
- augmentation uses advancing deterministic RNG streams;
- data-loader order does not depend on cache warmness;
- stratified batches do not sample the same raw row twice unless unavoidable;
- exact or semantic duplicate targets are measured within batches;
- each objective receives identical batch composition and sample count.

Duplicate targets are especially important for one-positive contrastive losses.
Freeze the equivalence predicate before pilots: prefer a canonical target ID or
exact canonical-caption hash; use semantic equivalence only with a
preregistered labeling rule. Hash the predicate/version and never derive
positives from the model embeddings under test. If equivalent captions occur,
use that frozen multi-positive mask or at minimum report the false-negative
rate and evaluate duplicate-aware retrieval.

### 3. Prove the primary metric is informative

Run controls before preregistering a probe as primary:

- seed-matched untrained features;
- shuffled labels;
- constant or raw-feature baseline;
- lower-capacity probe;
- subject- and dataset-balanced aggregation.

If a random high-dimensional representation nearly saturates a linear probe,
demote that probe to secondary. Prefer metrics tied to intended consumption,
such as duplicate-aware Recall@K, median rank, alignment gap, or a genuinely
held-out downstream task.

Do not treat adaptive-bandwidth HSIC/CKA as representation quality by default.
First evaluate it on synthetic collapsed, low-rank, spread, aligned, and
permuted embeddings.

### 4. Preregister a two-stage campaign

Commit before scientific runs:

- one primary prediction and quantitative magnitude;
- secondary diagnostic directions;
- absolute collapse and resource gates;
- pilot seed, steps, grids, and selection tie-breaker;
- confirmation seeds and paired statistical test;
- locked test policy;
- OOM and peak-VRAM ceiling;
- exact conditions that stop confirmation.

Run all pilots before choosing any configuration. Freeze an immutable
confirmation lock only when every required arm has an eligible selection.
Never manufacture or relax a lock after observing pilot results.

### 5. Implement against the evaluated representation

Trace the gradient from every regularizer to the exact tensor used downstream.

- A learned loss-only projection can absorb variance, decorrelation, or CCA
  constraints while the native representation collapses.
- Either evaluate the projected representation or regularize the native one.
- When feature dimension greatly exceeds batch size, avoid materializing a
  full covariance matrix. Use the exact batch-Gram identity when appropriate.
- Apply duplicate equivalence to the training target, not only evaluation.
- Keep objective components separately logged; total loss alone is ambiguous.

Measure collapse with at least two complementary diagnostics:

- median pairwise cosine for directional collapse;
- effective rank or spectral entropy for anisotropy.

Low cosine with low rank is concentrated low-dimensional geometry, not the
same failure as cosine near one.

### 6. Separate smoke from science

Run a 1–2 step engineering smoke before opening scientific artifacts. Verify:

- intended interpreter, device, and precision;
- finite total and component losses;
- gradient path and state persistence;
- checkpoint loadability;
- source and implementation hashes;
- peak allocated and reserved VRAM;
- no overlapping GPU process.

Tag smoke reports so aggregation excludes them. After the first scientific
run, do not patch the frozen implementation. Put any follow-up in a new output
root with a new manifest and hypothesis.

### 7. Serialize GPU experiments

When the user requests subagents or the machine has one GPU:

1. Give one bounded arm/grid to one subagent.
2. Require an idle-GPU preflight.
3. Run configurations sequentially inside that agent.
4. Preserve failed scientific artifacts.
5. Wait for process exit and integrity checks before starting the next agent.

Never use parallel GPU subagents merely to reduce wall-clock time.

Before each alternative, diff its fully resolved configuration against the
incumbent. Permit only the preregistered objective parameters and required
state/output fields to differ.

### 8. Apply the gate literally

After pilots:

- aggregate every planned candidate, including negative and collapsed runs;
- select only noncollapsed candidates;
- stop if no candidate satisfies every go condition;
- keep the locked test untouched after a stop;
- record the negative result and diagnose mechanism before proposing a
  follow-up.

A near miss is useful mechanistic evidence but is not a pass. Do not weaken an
existing gate. A revised threshold requires a new hypothesis and campaign.

### 9. Interpret objective families carefully

- **DCL/InfoNCE**: audit false negatives and uniformity; retrieval gains can
  coexist with low effective rank.
- **HSIC/kernel rewards**: confirm the measured dependence diagnostic actually
  rises; otherwise any gain may be generic regularization or selection noise.
- **SoftCCA/VICReg with projections**: inspect native geometry for loss-head
  bypass.
- **Native KTA/VICReg**: weak regularization may help alignment; stronger
  per-feature constraints can be counterproductive when dimension is much
  larger than batch size.
- **Multi-positive DCL**: prefer it as the first follow-up when duplicate
  captions create known false negatives.
- **Spectral regularization**: use a batch-Gram rank/entropy objective when the
  diagnosed failure is anisotropy rather than directional collapse.

Test one mechanistic change at a time.

## Required artifacts

Preserve:

- source, split, corrected-mask, and implementation hashes;
- complete resolved configuration;
- resolved-config diff against the incumbent and its allowlist;
- duplicate-equivalence predicate/version hash;
- environment/GPU report;
- initial and final checkpoints;
- per-step component log;
- predictions with stable evaluation indices;
- per-run report plus checksum;
- pilot aggregate and confirmation lock, if earned;
- a final report separating observations, inference, and untested hypotheses.

## Handoff checklist

- [ ] Untrained control and incumbent were both measured.
- [ ] Primary metric passed informativeness controls.
- [ ] Duplicate positives and sampler duplication were audited.
- [ ] Native, not merely projected, geometry was measured.
- [ ] Smoke runs cannot enter scientific aggregation.
- [ ] Pilots ran before selection and without GPU overlap.
- [ ] Failed gates stopped confirmation.
- [ ] Direction-only tool verdicts were not confused with magnitude gates.
- [ ] Population preprocessing did not see held-out entities.
- [ ] Locked test remained untouched unless confirmation was earned.
