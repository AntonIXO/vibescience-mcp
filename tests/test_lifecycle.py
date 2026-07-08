"""End-to-end lifecycle + gate tests (brief §12 acceptance)."""

import pytest

from vibescience_mcp.core import Store, VibeScienceError
from vibescience_mcp.models import HypothesisStatus, Verdict


def test_full_loop_empty_vault(vault):
    """create_problem → recall(empty) → register → propose(rejects if no pred)
    → start(auto git_ref) → record → close(auto verdict)."""
    s = Store(vault)

    s.create_problem("test problem", id="p1", topic_tags=["t"], problem_tags=["pt"])

    # recall on an empty vault returns nothing
    r = s.recall(problem_id="p1")
    assert r["results"] == []

    s.register_diagnostic("metric a", direction="higher_better", id="a")
    s.register_intervention("do thing", id="do-thing")

    # propose without a prediction is rejected
    with pytest.raises(VibeScienceError, match="predicted_effect"):
        s.propose_hypothesis("p1", "no prediction here", predicted_effects=[])

    h = s.propose_hypothesis(
        "p1", "doing the thing raises a", id="h1",
        interventions=["do-thing"],
        predicted_effects=[{"diagnostic_id": "a", "direction": "up"}],
    )
    assert h.status == HypothesisStatus.proposed

    e = s.start_experiment("h1", id="e1")  # no git_ref passed
    # git_ref is auto-filled (may be empty if not a repo, but attribute exists)
    assert e.hypothesis_id == "h1"
    assert s.get_hypothesis("h1").status == HypothesisStatus.testing

    s.record_diagnostics("e1", [{"diagnostic_id": "a", "before": 0.1, "after": 0.5}])
    res = s.close_experiment("e1")
    assert res["verdict"] == "supports"
    assert res["prediction_match"]["overall"] is True
    assert s.get_hypothesis("h1").status == HypothesisStatus.confirmed
    assert "committing" in res["suggested_next_action"]


def test_propose_rejects_unregistered_diagnostic(vault):
    s = Store(vault)
    s.create_problem("p", id="p1")
    with pytest.raises(VibeScienceError, match="unregistered diagnostic"):
        s.propose_hypothesis("p1", "stmt",
                             predicted_effects=[{"diagnostic_id": "ghost", "direction": "up"}])


def test_propose_rejects_unknown_intervention(vault):
    s = Store(vault)
    s.create_problem("p", id="p1")
    s.register_diagnostic("a", id="a", direction="higher_better")
    with pytest.raises(VibeScienceError, match="Unknown intervention"):
        s.propose_hypothesis("p1", "stmt", interventions=["ghost"],
                             predicted_effects=[{"diagnostic_id": "a", "direction": "up"}])


def test_close_requires_measurements(vault):
    s = Store(vault)
    s.create_problem("p", id="p1")
    s.register_diagnostic("a", id="a", direction="higher_better")
    s.propose_hypothesis("p1", "stmt", id="h1",
                         predicted_effects=[{"diagnostic_id": "a", "direction": "up"}])
    s.start_experiment("h1", id="e1")
    with pytest.raises(VibeScienceError, match="no diagnostics"):
        s.close_experiment("e1")


def test_seeded_verdicts(seeded):
    """The §13 fixture: confirming hyp supports, refuted sibling refutes."""
    conf = seeded.get_hypothesis("projector-freeze-blocks-caption-grad")
    refu = seeded.get_hypothesis("rmsnorm-logscale-alone-fixes-collapse")
    assert conf.status == HypothesisStatus.confirmed
    assert refu.status == HypothesisStatus.refuted

    e_conf = seeded.vault.read_experiment("unfreeze-projector-joint-loss")
    assert e_conf.verdict == Verdict.supports
    assert e_conf.prediction_match.overall is True

    e_refu = seeded.vault.read_experiment("rmsnorm-logscale-alone")
    assert e_refu.verdict == Verdict.refutes
    assert e_refu.prediction_match.overall is False
