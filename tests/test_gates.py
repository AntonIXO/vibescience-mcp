"""Multi-level verdicts: direction is not the same as the preregistered gate.

Driven by a real failure on the GPU host. An EiV confirmation run recorded
`verdict: supports` while its own preregistered superiority gate had failed:

    superiority_passed: False       exact_one_sided_signflip_p: 0.34375
    paired bootstrap CI95: [-0.00848, +0.01347]   (crosses zero)
    paired ΔRecall@5 signs: -, +, +, -, +          (noise)

The verdict was formally correct — mean ΔRecall@5 = +0.003 > 0, direction `up`
predicted — and scientifically misleading. The agent knew, and compensated in
free-text notes ("Magnitude/confirmation gate failed despite any direction-only
computed verdict"). But calibration/causal_map/recall read only the verdict
field, so the machine-readable layer asserted a win the gate had rejected.

The host's own `curate-long-running-experiments` skill §8 already mandates the
distinction: "directional support; preregistered magnitude-gate pass;
deployment/collapse-gate pass; locked-test eligibility. Do not equate one with
another."
"""

import pytest

from vibescience_mcp.models import (
    Direction,
    Gate,
    GateResult,
    ObservedEffect,
    PredictedEffect,
    Verdict,
    compute_verdict,
)


def test_direction_right_but_blocking_gate_failed_is_not_supports():
    """The exact EiV v3 shape: right direction, failed magnitude gate."""
    pred = [PredictedEffect(diagnostic_id="recall5", direction=Direction.up)]
    obs = [ObservedEffect(diagnostic_id="recall5", delta=+0.003, direction=Direction.up)]
    gates = [Gate(id="mean_recall_delta", description="ΔRecall@5 >= +0.010", blocking=True)]
    results = [GateResult(gate_id="mean_recall_delta", passed=False,
                          evidence="delta=+0.00299 < 0.010; signflip p=0.34375")]
    assert compute_verdict(pred, obs, gates, results) == Verdict.directional_only


def test_direction_right_and_gates_passed_is_supports():
    pred = [PredictedEffect(diagnostic_id="recall5", direction=Direction.up)]
    obs = [ObservedEffect(diagnostic_id="recall5", delta=+0.02, direction=Direction.up)]
    gates = [Gate(id="mean_recall_delta", blocking=True)]
    results = [GateResult(gate_id="mean_recall_delta", passed=True)]
    assert compute_verdict(pred, obs, gates, results) == Verdict.supports


def test_no_gates_declared_behaves_exactly_as_before():
    """Backward compatibility: 16 live hypotheses on the GPU host have no gates."""
    pred = [PredictedEffect(diagnostic_id="d", direction=Direction.up)]
    obs = [ObservedEffect(diagnostic_id="d", delta=+1.0, direction=Direction.up)]
    assert compute_verdict(pred, obs) == Verdict.supports
    assert compute_verdict(pred, obs, [], []) == Verdict.supports


def test_non_blocking_gate_failure_does_not_downgrade():
    """An advisory gate records information without vetoing the claim."""
    pred = [PredictedEffect(diagnostic_id="d", direction=Direction.up)]
    obs = [ObservedEffect(diagnostic_id="d", delta=+1.0, direction=Direction.up)]
    gates = [Gate(id="nice_to_have", blocking=False)]
    results = [GateResult(gate_id="nice_to_have", passed=False)]
    assert compute_verdict(pred, obs, gates, results) == Verdict.supports


def test_wrong_direction_still_refutes_regardless_of_gates():
    """Gates can only downgrade a win, never rescue a refutation."""
    pred = [PredictedEffect(diagnostic_id="d", direction=Direction.up)]
    obs = [ObservedEffect(diagnostic_id="d", delta=-1.0, direction=Direction.down)]
    gates = [Gate(id="g", blocking=True)]
    results = [GateResult(gate_id="g", passed=True)]
    assert compute_verdict(pred, obs, gates, results) == Verdict.refutes


def test_unreported_blocking_gate_counts_as_not_passed():
    """Silence is not a pass."""
    pred = [PredictedEffect(diagnostic_id="d", direction=Direction.up)]
    obs = [ObservedEffect(diagnostic_id="d", delta=+1.0, direction=Direction.up)]
    gates = [Gate(id="never_reported", blocking=True)]
    assert compute_verdict(pred, obs, gates, []) == Verdict.directional_only


def test_one_failed_gate_among_many_is_enough_to_downgrade():
    pred = [PredictedEffect(diagnostic_id="d", direction=Direction.up)]
    obs = [ObservedEffect(diagnostic_id="d", delta=+1.0, direction=Direction.up)]
    gates = [Gate(id="a", blocking=True), Gate(id="b", blocking=True),
             Gate(id="c", blocking=True)]
    results = [GateResult(gate_id="a", passed=True),
               GateResult(gate_id="b", passed=False),
               GateResult(gate_id="c", passed=True)]
    assert compute_verdict(pred, obs, gates, results) == Verdict.directional_only


def test_unmoved_diagnostic_stays_inconclusive_even_with_passing_gates():
    pred = [PredictedEffect(diagnostic_id="d", direction=Direction.up)]
    obs = [ObservedEffect(diagnostic_id="d", delta=0.0, direction=Direction.none)]
    gates = [Gate(id="g", blocking=True)]
    results = [GateResult(gate_id="g", passed=True)]
    assert compute_verdict(pred, obs, gates, results) == Verdict.inconclusive


def test_directional_only_maps_to_an_unproven_hypothesis():
    """A downgraded verdict must NOT mark the hypothesis confirmed."""
    from vibescience_mcp.models import HypothesisStatus, verdict_to_hypothesis_status
    st = verdict_to_hypothesis_status(Verdict.directional_only)
    assert st != HypothesisStatus.confirmed
    assert st == HypothesisStatus.inconclusive
