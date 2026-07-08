"""Unit tests for the verdict / prediction_match computation (brief §5, §12)."""

from vibescience_mcp.models import (
    Direction,
    PredictedEffect,
    Measurement,
    Verdict,
    compute_observed_effects,
    compute_prediction_match,
    compute_verdict,
    verdict_to_hypothesis_status,
    HypothesisStatus,
)


def pe(diag, direction):
    return PredictedEffect(diagnostic_id=diag, direction=Direction(direction))


def meas(diag, before, after):
    return Measurement(diagnostic_id=diag, before=before, after=after)


def test_observed_direction_up_down_flat():
    obs = compute_observed_effects([
        meas("a", 1.0, 2.0),   # up
        meas("b", 5.0, 1.0),   # down
        meas("c", 3.0, 3.0),   # flat
    ])
    d = {o.diagnostic_id: o for o in obs}
    assert d["a"].direction == Direction.up and d["a"].delta == 1.0
    assert d["b"].direction == Direction.down and d["b"].delta == -4.0
    assert d["c"].direction == Direction.none and d["c"].delta == 0.0


def test_verdict_supports_when_primary_matches():
    pred = [pe("attn", "up"), pe("hr", "down")]
    obs = compute_observed_effects([meas("attn", 0.02, 0.31), meas("hr", 8.1, 4.0)])
    assert compute_verdict(pred, obs) == Verdict.supports
    pm = compute_prediction_match(pred, obs)
    assert pm.overall is True
    assert pm.per_diagnostic == {"attn": True, "hr": True}


def test_verdict_refutes_when_primary_goes_wrong_way():
    pred = [pe("attn", "up")]
    obs = compute_observed_effects([meas("attn", 0.02, 0.015)])  # went down
    assert compute_verdict(pred, obs) == Verdict.refutes
    assert compute_prediction_match(pred, obs).overall is False


def test_verdict_inconclusive_when_primary_flat():
    pred = [pe("attn", "up")]
    obs = compute_observed_effects([meas("attn", 0.2, 0.2)])  # no move
    assert compute_verdict(pred, obs) == Verdict.inconclusive


def test_verdict_inconclusive_when_primary_not_measured():
    pred = [pe("attn", "up")]
    obs = compute_observed_effects([meas("other", 1.0, 2.0)])
    assert compute_verdict(pred, obs) == Verdict.inconclusive


def test_overall_ignores_secondary_mismatch():
    # primary matches, secondary doesn't → still supports/overall True
    pred = [pe("attn", "up"), pe("hr", "down")]
    obs = compute_observed_effects([meas("attn", 0.0, 0.3), meas("hr", 4.0, 8.0)])  # hr up (wrong)
    assert compute_verdict(pred, obs) == Verdict.supports
    pm = compute_prediction_match(pred, obs)
    assert pm.overall is True
    assert pm.per_diagnostic == {"attn": True, "hr": False}


def test_null_prediction_refuted_if_it_moves():
    pred = [pe("attn", "none")]
    obs = compute_observed_effects([meas("attn", 0.1, 0.5)])  # predicted no move, it moved
    assert compute_verdict(pred, obs) == Verdict.refutes


def test_verdict_to_status_mapping():
    assert verdict_to_hypothesis_status(Verdict.supports) == HypothesisStatus.confirmed
    assert verdict_to_hypothesis_status(Verdict.refutes) == HypothesisStatus.refuted
    assert verdict_to_hypothesis_status(Verdict.inconclusive) == HypothesisStatus.inconclusive
