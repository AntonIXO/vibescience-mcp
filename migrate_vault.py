"""Migrate an existing vault to the registered-tag vocabulary + DAG schema.

Lossless and idempotent. It does NOT silently merge tags: collapsing e.g.
`causality` (CCM / directional coupling) into `causal-inference` (g-formula /
potential outcomes) would destroy a real scientific distinction. Merge
candidates are REPORTED for a human decision instead — the same principle the
Anthropic KG cookbook applies to entity resolution (additive and reversible).

What it does do:
  1. registers every tag already in use, on the axis it is already used on;
  2. re-writes hypotheses/experiments so tags are INHERITED down
     problem -> hypothesis -> experiment, which is what turns a tag from a
     label on one node into an actual edge;
  3. reports orphans (n_uses<=1 or confined to a single entity kind) and
     near-duplicate pairs.

Usage:  python migrate_vault.py <vault_path> [--apply]
"""

from __future__ import annotations

import difflib
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from vibescience_mcp.core import Store  # noqa: E402


def collect(store: Store) -> dict[str, Counter]:
    """tag -> Counter(axis) as actually used across the vault today."""
    seen: dict[str, Counter] = defaultdict(Counter)
    for p in store.vault.all_problems():
        for t in p.topic_tags:
            seen[t]["topic"] += 1
        for t in p.problem_tags:
            seen[t]["problem"] += 1
    for h in store.vault.all_hypotheses():
        for t in h.topic_tags:
            seen[t]["topic"] += 1
        for t in h.problem_tags:
            seen[t]["problem"] += 1
    for iv in store.vault.all_interventions():
        for t in iv.topic_tags:
            seen[t]["topic"] += 1
    for pa in store.vault.all_papers():
        for t in pa.topic_tags:
            seen[t]["topic"] += 1
    return seen


def main(vault_path: str, apply: bool) -> int:
    store = Store(vault_path)
    seen = collect(store)
    print(f"vault: {vault_path}")
    print(f"distinct tags in use: {len(seen)}")

    conflicts = {t: dict(c) for t, c in seen.items() if len(c) > 1}
    if conflicts:
        print("\n!! tags used on BOTH axes (pick one axis manually):")
        for t, c in conflicts.items():
            print(f"   {t}: {c}")
        print("   aborting — resolve these first.")
        return 1

    if not apply:
        print("\n--- DRY RUN (pass --apply to write) ---")

    # 1. register the vocabulary as it already exists
    registered = 0
    for tag, c in sorted(seen.items()):
        axis = next(iter(c))
        if apply:
            store.register_tag(tag, axis=axis)
        registered += 1
    print(f"\n1. registered {registered} tags")

    # 2. re-write so tags propagate down the chain (this creates the edges)
    if apply:
        for h in store.vault.all_hypotheses():
            h.topic_tags = store._inherit_tags(
                h.problem_id, h.topic_tags, h.interventions, "topic")
            h.problem_tags = store._inherit_tags(
                h.problem_id, h.problem_tags, None, "problem")
            store.vault.write_hypothesis(h)
        hyp = {h.id: h for h in store.vault.all_hypotheses()}
        for e in store.vault.all_experiments():
            h = hyp.get(e.hypothesis_id)
            pred = ({pe.diagnostic_id: pe.direction for pe in h.predicted_effects}
                    if h else None)
            store.vault.write_experiment(
                e, predicted=pred,
                tags=(h.topic_tags + h.problem_tags) if h else None)
        store.reindex()
        print("2. propagated tags problem -> hypothesis -> experiment")
    else:
        print("2. would propagate tags problem -> hypothesis -> experiment")

    # 3. report what is still dead weight
    if apply:
        rows = store.list_tags()
        orphans = [r for r in rows if r["orphan"]]
        print(f"\n3. orphan tags remaining: {len(orphans)}/{len(rows)}")
        for r in orphans:
            print(f"   {r['id']:28} n={r['n_uses']} kinds={r['kinds']}")

        ids = [r["id"] for r in rows]
        pairs, done = [], set()
        for t in ids:
            for near in difflib.get_close_matches(t, ids, n=3, cutoff=0.72):
                key = tuple(sorted((t, near)))
                if near != t and key not in done:
                    done.add(key)
                    pairs.append(key)
        if pairs:
            print("\n   near-duplicate candidates (NOT merged — your call):")
            for a, b in pairs:
                print(f"   {a}  <->  {b}")
            print("   to merge: register_tag(id='<keep>', axis=..., aliases=['<drop>'])")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], "--apply" in sys.argv))
