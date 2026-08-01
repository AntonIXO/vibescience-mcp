"""Tag registry, experiment DAG, and the crashed verdict.

These cover the three defects found by auditing the live vault:
  1. tags were free text -> 18/36 orphans, 20/36 confined to one entity kind,
     and `recall` only queried `hypotheses` so 69% of tag mass was unreachable;
  2. experiments were a flat list -> no lineage / frontier;
  3. a crashed run had nowhere to go -> it either stayed open forever or got
     faked as a null result, poisoning calibration.
"""

import pytest

from vibescience_mcp.core import Store, VibeScienceError
from vibescience_mcp.models import HypothesisStatus, Verdict


# --------------------------------------------------------------------------- #
# Tag registry
# --------------------------------------------------------------------------- #
def test_unknown_tag_is_rejected_with_suggestion(vault):
    s = Store(vault)
    s.register_tag("robust-covariance", axis="topic")

    with pytest.raises(VibeScienceError) as ei:
        s.create_problem("p", id="p", topic_tags=["robust-covarience"])  # typo
    msg = str(ei.value)
    assert "Unknown topic tag" in msg
    assert "robust-covariance" in msg  # near-match suggestion


def test_alias_resolves_to_canonical_tag(vault):
    s = Store(vault)
    s.register_tag("causal-inference", axis="topic", aliases=["causality"])
    p = s.create_problem("p", id="p", topic_tags=["causality"])
    assert p.topic_tags == ["causal-inference"], "alias must collapse to canonical"


def test_axis_is_enforced(vault):
    s = Store(vault)
    s.register_tag("masking", axis="problem")
    with pytest.raises(VibeScienceError, match="registered on axis 'problem'"):
        s.create_problem("p", id="p", topic_tags=["masking"])


def test_alias_cannot_be_hijacked_by_another_tag(vault):
    s = Store(vault)
    s.register_tag("causal-inference", axis="topic", aliases=["causality"])
    s.register_tag("coupling", axis="topic")
    with pytest.raises(VibeScienceError, match="already belongs to tag"):
        s.register_tag("coupling", axis="topic", aliases=["causality"])


def test_register_tag_is_idempotent_and_merges_aliases(vault):
    s = Store(vault)
    s.register_tag("circadian", axis="topic", aliases=["rhythm"])
    t = s.register_tag("circadian", axis="topic", aliases=["rest-activity"])
    assert set(t.aliases) == {"rhythm", "rest-activity"}, "aliases merge, never drop"


def test_hypothesis_inherits_tags_from_problem_and_intervention(vault):
    """The root cause of orphan tags: every entity was tagged by hand."""
    s = Store(vault)
    s.register_tag("anomaly-detection", axis="topic")
    s.register_tag("resting-hr", axis="topic")
    s.register_tag("missing-early-warning", axis="problem")
    s.register_diagnostic("yield", direction="higher_better", id="y")
    s.create_problem("p", id="p", topic_tags=["anomaly-detection"],
                     problem_tags=["missing-early-warning"])
    s.register_intervention("cusum", id="cusum", topic_tags=["resting-hr"])

    h = s.propose_hypothesis("p", "cusum surfaces early warning", id="h",
                             interventions=["cusum"],
                             predicted_effects=[{"diagnostic_id": "y", "direction": "up"}])
    assert set(h.topic_tags) == {"anomaly-detection", "resting-hr"}
    assert h.problem_tags == ["missing-early-warning"]


def test_list_tags_flags_orphans(vault):
    s = Store(vault)
    s.register_tag("used", axis="topic")
    s.register_tag("never-used", axis="topic")
    s.create_problem("p", id="p", topic_tags=["used"])

    by_id = {t["id"]: t for t in s.list_tags()}
    assert by_id["never-used"]["n_uses"] == 0
    assert by_id["never-used"]["orphan"] is True
    # 'used' sits on exactly one entity kind -> still an orphan edge
    assert by_id["used"]["orphan"] is True


def test_recall_reaches_papers_and_interventions(seeded):
    """Regression: tags on papers/interventions used to be invisible to recall."""
    r = seeded.recall(topic_tags=["blip2-qformer"])
    kinds = {c["kind"] for c in r["context"]}
    assert "paper" in kinds, "a tag matching only papers must not return nothing"
    ids = {c["id"] for c in r["context"]}
    assert "blip2" in ids


def test_recall_resolves_alias_in_query(seeded):
    """Querying by a synonym must still hit — that was the silent-miss bug."""
    canonical = seeded.recall(problem_tags=["projector-freeze"])
    via_alias = seeded.recall(problem_tags=["frozen-projector"])
    assert via_alias["results"], "alias query must return results"
    assert ([x["hypothesis_id"] for x in canonical["results"]]
            == [x["hypothesis_id"] for x in via_alias["results"]])


def test_experiment_is_tag_searchable(seeded):
    conn = seeded._conn()
    n = conn.execute(
        "SELECT COUNT(*) FROM tags WHERE entity_kind='experiment'"
    ).fetchone()[0]
    conn.close()
    assert n > 0, "experiments must inherit hypothesis tags into the index"


# --------------------------------------------------------------------------- #
# Experiment DAG
# --------------------------------------------------------------------------- #
def _dag_vault(vault):
    s = Store(vault)
    s.register_diagnostic("metric", direction="higher_better", id="m")
    s.create_problem("p", id="p")
    s.propose_hypothesis("p", "m goes up", id="h",
                         predicted_effects=[{"diagnostic_id": "m", "direction": "up"}])
    return s


def test_lineage_children_leaves(vault):
    s = _dag_vault(vault)
    s.start_experiment("h", id="e1")
    s.record_diagnostics("e1", [{"diagnostic_id": "m", "before": 0.1, "after": 0.2}])
    s.close_experiment("e1")
    s.start_experiment("h", id="e2", parent_experiment_id="e1")
    s.start_experiment("h", id="e3", parent_experiment_id="e2")
    s.start_experiment("h", id="e4", parent_experiment_id="e1")  # sibling branch

    assert [n["experiment_id"] for n in s.lineage("e3")["path"]] == ["e1", "e2", "e3"]
    assert s.lineage("e3")["depth"] == 3

    kids = {c["experiment_id"] for c in s.children("e1")["children"]}
    assert kids == {"e2", "e4"}
    assert s.children("e3")["n_children"] == 0

    leaf_ids = {leaf["experiment_id"] for leaf in s.leaves()["leaves"]}
    assert leaf_ids == {"e3", "e4"}, "frontier = experiments with no children"

    # e1 is closed; e3/e4 are the dangling ones
    open_leaves = {leaf["experiment_id"] for leaf in s.leaves(unevaluated_only=True)["leaves"]}
    assert open_leaves == {"e3", "e4"}


def test_unknown_parent_is_rejected(vault):
    s = _dag_vault(vault)
    with pytest.raises(VibeScienceError, match="Unknown parent_experiment_id"):
        s.start_experiment("h", id="e1", parent_experiment_id="nope")


def test_lineage_survives_reindex(vault):
    s = _dag_vault(vault)
    s.start_experiment("h", id="e1")
    s.start_experiment("h", id="e2", parent_experiment_id="e1")
    before = s.lineage("e2")
    s.db_path.unlink()
    s.reindex()
    assert s.lineage("e2") == before, "markdown is truth; the index rebuilds identically"


# --------------------------------------------------------------------------- #
# Crashed runs
# --------------------------------------------------------------------------- #
def test_crash_is_not_evidence(vault):
    s = _dag_vault(vault)
    s.start_experiment("h", id="e1")
    out = s.abort_experiment("e1", crash_reason="CUDA OOM at step 40")

    assert out["verdict"] == "crashed"
    # the hypothesis stays retryable — a crash is not a refutation
    assert s.get_hypothesis("h").status == HypothesisStatus.testing
    assert s.get_hypothesis("h").status != HypothesisStatus.refuted

    # and it must never reach calibration
    assert s.calibration()["n"] == 0
    conn = s._conn()
    assert conn.execute("SELECT COUNT(*) FROM evaluation").fetchone()[0] == 0
    conn.close()


def test_crashed_experiment_is_closed_and_visible_in_dag(vault):
    s = _dag_vault(vault)
    s.start_experiment("h", id="e1")
    s.abort_experiment("e1", crash_reason="boom")
    e = s.vault.read_experiment("e1")
    assert e.closed is True and e.verdict == Verdict.crashed
    assert e.crash_reason == "boom"
    # closed, so not dangling, but still a leaf on the frontier
    assert {x["experiment_id"] for x in s.leaves()["leaves"]} == {"e1"}
    assert s.leaves(unevaluated_only=True)["leaves"] == []


def test_abort_requires_a_reason_and_rejects_double_close(vault):
    s = _dag_vault(vault)
    s.start_experiment("h", id="e1")
    with pytest.raises(VibeScienceError, match="crash_reason"):
        s.abort_experiment("e1", crash_reason="")
    s.abort_experiment("e1", crash_reason="boom")
    with pytest.raises(VibeScienceError, match="already closed"):
        s.abort_experiment("e1", crash_reason="again")


# --------------------------------------------------------------------------- #
# Regression: prod crash found in Codex rollouts (2026-07-28)
# --------------------------------------------------------------------------- #
def test_recall_survives_unmeasured_primary_diagnostic(vault):
    """`recall` crashed 5x in a real Codex run with
    `unsupported format string passed to NoneType.__format__`.

    Cause: a hypothesis predicts a PRIMARY diagnostic that the experiment never
    measured, so evaluation.observed/delta are NULL and f"{None:+g}" raised.
    Worst possible blast radius: it only fired on refuted/inconclusive rows —
    exactly the negative results the pre-mortem gate exists to surface — so the
    tool died precisely when it mattered.
    """
    s = Store(vault)
    s.register_tag("t", axis="topic")
    s.register_diagnostic("primary", direction="higher_better", id="prim")
    s.register_diagnostic("other", direction="higher_better", id="other")
    s.register_intervention("iv", id="iv", topic_tags=["t"])
    s.create_problem("p", id="p", topic_tags=["t"])
    s.propose_hypothesis(
        "p", "prim goes up", id="h", topic_tags=["t"], interventions=["iv"],
        predicted_effects=[{"diagnostic_id": "prim", "direction": "up"},
                           {"diagnostic_id": "other", "direction": "up"}])
    s.start_experiment("h", id="e")
    # only the SECONDARY diagnostic gets measured
    s.record_diagnostics("e", [{"diagnostic_id": "other", "before": 1.0, "after": 2.0}])
    assert s.close_experiment("e")["verdict"] == "inconclusive"

    r = s.recall(topic_tags=["t"])          # used to raise TypeError
    why = r["results"][0]["why_it_failed"]
    assert "never" in why and "measured" in why, why

    # the sibling surfaces must not trip on the same NULL either
    s.causal_map(tag="t")
    s.calibration(tag="t")
    s.write_canvas(tag="t")
