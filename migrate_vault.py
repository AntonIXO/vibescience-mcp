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
    """tag -> Counter(axis) as actually used today, WEIGHTED by entity kind.

    Problems and hypotheses carry the scientific meaning of a tag; papers and
    interventions merely reference it. So a problem/hypothesis vote counts
    double. This makes axis resolution deterministic on real data instead of
    tie-breaking arbitrarily (`false-negatives` was a literal 3-3 split, and it
    is unambiguously a failure mode, i.e. a problem tag).
    """
    W_PRIMARY, W_SECONDARY = 2, 1
    seen: dict[str, Counter] = defaultdict(Counter)
    for p in store.vault.all_problems():
        for t in p.topic_tags:
            seen[t]["topic"] += W_PRIMARY
        for t in p.problem_tags:
            seen[t]["problem"] += W_PRIMARY
    for h in store.vault.all_hypotheses():
        for t in h.topic_tags:
            seen[t]["topic"] += W_PRIMARY
        for t in h.problem_tags:
            seen[t]["problem"] += W_PRIMARY
    for iv in store.vault.all_interventions():
        for t in iv.topic_tags:
            seen[t]["topic"] += W_SECONDARY
    for pa in store.vault.all_papers():
        for t in pa.topic_tags:
            seen[t]["topic"] += W_SECONDARY
    return seen


def main(vault_path: str, apply: bool, prune: bool = False) -> int:
    store = Store(vault_path)
    seen = collect(store)
    print(f"vault: {vault_path}")
    print(f"distinct tags in use: {len(seen)}")

    # A tag used on both axes is ambiguous. Resolve by majority: problems and
    # hypotheses carry the scientific meaning, so whichever axis they used wins.
    # Ties abort — a genuine 50/50 needs a human.
    conflicts = {t: dict(c) for t, c in seen.items() if len(c) > 1}
    resolved: dict[str, str] = {}
    if conflicts:
        print("\n!! tags used on BOTH axes — resolving by majority:")
        for t, c in conflicts.items():
            top = sorted(c.items(), key=lambda kv: -kv[1])
            if top[0][1] == top[1][1]:
                print(f"   {t}: {c}  <<< TIE, cannot resolve automatically")
                print("   aborting — pick an axis for this tag by hand.")
                return 1
            resolved[t] = top[0][0]
            print(f"   {t}: {c} -> '{top[0][0]}'")

    if not apply:
        print("\n--- DRY RUN (pass --apply to write) ---")

    # 1. register the vocabulary as it already exists
    registered = 0
    for tag, c in sorted(seen.items()):
        axis = resolved.get(tag) or next(iter(c))
        if apply:
            store.register_tag(tag, axis=axis)
        registered += 1
    print(f"\n1. registered {registered} tags")

    # 2. re-write so tags propagate down the chain (this creates the edges).
    #    Ambiguous tags are moved onto their resolved axis on the way through.
    if apply:
        def split_axes(tags):
            """Return (topic, problem) after applying the axis resolution."""
            topic, problem = [], []
            for t in tags:
                ax = resolved.get(t)
                if ax == "problem":
                    problem.append(t)
                elif ax == "topic":
                    topic.append(t)
                else:
                    topic.append(t)  # caller decides; unresolved keeps its slot
            return topic, problem

        for p in store.vault.all_problems():
            keep_t, moved_p = split_axes(p.topic_tags)
            keep_p, moved_t = ([t for t in p.problem_tags
                                if resolved.get(t, "problem") == "problem"],
                               [t for t in p.problem_tags
                                if resolved.get(t) == "topic"])
            p.topic_tags = list(dict.fromkeys(keep_t + moved_t))
            p.problem_tags = list(dict.fromkeys(keep_p + moved_p))
            store.vault.write_problem(p)

        for h in store.vault.all_hypotheses():
            keep_t, moved_p = split_axes(h.topic_tags)
            keep_p = [t for t in h.problem_tags if resolved.get(t, "problem") == "problem"]
            moved_t = [t for t in h.problem_tags if resolved.get(t) == "topic"]
            h.topic_tags = store._inherit_tags(
                h.problem_id, list(dict.fromkeys(keep_t + moved_t)),
                h.interventions, "topic")
            h.problem_tags = store._inherit_tags(
                h.problem_id, list(dict.fromkeys(keep_p + moved_p)), None, "problem")
            store.vault.write_hypothesis(h)

        for iv in store.vault.all_interventions():
            keep_t = [t for t in iv.topic_tags if resolved.get(t, "topic") == "topic"]
            if keep_t != iv.topic_tags:
                iv.topic_tags = keep_t
                store.vault.write_intervention(iv)
        for pa in store.vault.all_papers():
            keep_t = [t for t in pa.topic_tags if resolved.get(t, "topic") == "topic"]
            if keep_t != pa.topic_tags:
                pa.topic_tags = keep_t
                store.vault.write_paper(pa)

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

    # 3. report what is still dead weight, and optionally prune it
    if apply:
        rows = store.list_tags()
        orphans = [r for r in rows if r["orphan"]]
        print(f"\n3. orphan tags: {len(orphans)}/{len(rows)}")
        for r in orphans:
            print(f"   {r['id']:28} n={r['n_uses']} kinds={r['kinds']}")

        if prune and orphans:
            print(f"\n   PRUNING {len(orphans)} orphans (n<=1 or single entity kind)")
            drop = {r["id"] for r in orphans}
            for p in store.vault.all_problems():
                p.topic_tags = [t for t in p.topic_tags if t not in drop]
                p.problem_tags = [t for t in p.problem_tags if t not in drop]
                store.vault.write_problem(p)
            for h in store.vault.all_hypotheses():
                h.topic_tags = [t for t in h.topic_tags if t not in drop]
                h.problem_tags = [t for t in h.problem_tags if t not in drop]
                store.vault.write_hypothesis(h)
            for iv in store.vault.all_interventions():
                iv.topic_tags = [t for t in iv.topic_tags if t not in drop]
                store.vault.write_intervention(iv)
            for pa in store.vault.all_papers():
                pa.topic_tags = [t for t in pa.topic_tags if t not in drop]
                store.vault.write_paper(pa)
            hyp = {h.id: h for h in store.vault.all_hypotheses()}
            for e in store.vault.all_experiments():
                h = hyp.get(e.hypothesis_id)
                pred = ({pe.diagnostic_id: pe.direction for pe in h.predicted_effects}
                        if h else None)
                store.vault.write_experiment(
                    e, predicted=pred,
                    tags=(h.topic_tags + h.problem_tags) if h else None)
            for tid in drop:
                store.vault.path("tags", tid).unlink(missing_ok=True)
            store.reindex()
            left = store.list_tags()
            print(f"   vocabulary: {len(rows)} -> {len(left)} tags, "
                  f"{sum(1 for r in left if r['orphan'])} still orphaned")

        ids = [r["id"] for r in store.list_tags()]
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
    raise SystemExit(main(sys.argv[1], "--apply" in sys.argv, "--prune" in sys.argv))
