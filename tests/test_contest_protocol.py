"""The submitted tables must stay linked to the code and stored inputs."""
import hashlib
import json

import numpy as np
import pytest

from checks.contest_experiments import (PROTOCOL, SOURCE_PATHS, _validate_known_case,
                                       run_method, timed_case, high_precision_reference,
                                       reference_relative_error)
from checks.render_contest_tables import load_protocol, render_tables
from examples.cases import analytic_cases


def test_known_value_check_is_independent_of_success_status():
    case = analytic_cases()[1]
    result = run_method(case.A, case.p, "general")
    _validate_known_case(case, result)
    result["hyperbolic_values"][0] *= 1.01
    with pytest.raises(AssertionError):
        _validate_known_case(case, result)


def saved_protocol(tmp_path):
    """Minimal complete log with a shared-domain input and a wide input."""
    digests = {}
    for source in SOURCE_PATHS:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        digests[source] = hashlib.sha256(path.read_bytes()).hexdigest()
    np.savez_compressed(tmp_path / "inputs.npz", full=np.eye(2), wide=np.ones((1, 2)))
    metrics = dict(reconstruction_error=0., unitarity_error=0., j_unitarity_error=0.)
    record = dict(status="ok", metrics=metrics, elapsed_seconds=.001,
                  max_relative_value_error_stored_input=0., structure=dict(j=0))
    timed = dict(**record, time_seconds=[.001], median_ms=1.,
                 attempts=[dict(repetition=0, status="ok", elapsed_seconds=.001)])
    payload = dict(protocol=PROTOCOL, sha256=digests,
                   inputs_sha256=hashlib.sha256((tmp_path / "inputs.npz").read_bytes()).hexdigest(),
                   settings=dict(timing_repetitions=1),
                   structural=[dict(input_key="full", shape=[2,2], p=1, group="analytic",
                                    complex_data=False, jacobi=record, **record)],
                   comparison=[dict(input_key="full", shape=[2,2], p=1, general=timed, jacobi=timed)],
                   conditioning=[dict(input_key="full", shape=[2,2], p=1, sigma_ratio=1.,
                                      general=record, jacobi=record)],
                   thresholds=[dict(input_key="wide", shape=[1,2], p=1, delta=0., spectral_rtol=1e-14,
                                    jacobi=dict(status="not_applicable", elapsed_seconds=None,
                                                reason="Fewer rows than columns"), **record)])
    protocol = tmp_path / "results.json"
    protocol.write_text(json.dumps(payload))
    return protocol


@pytest.mark.parametrize("changed", ["hsvd/core.py", "inputs.npz", "checks/render_contest_tables.py"])
def test_changed_sources_or_inputs_cannot_produce_trusted_tables(tmp_path, changed):
    protocol = saved_protocol(tmp_path)
    load_protocol(protocol, tmp_path)
    (tmp_path / changed).write_bytes(b"modified")
    with pytest.raises(ValueError, match="Stale protocol|Saved inputs"):
        load_protocol(protocol, tmp_path)


def test_saved_matrix_rank_prevents_a_fabricated_jacobi_comparison(tmp_path):
    path = saved_protocol(tmp_path)
    payload = json.loads(path.read_text())
    payload["comparison"][0].update(input_key="wide", shape=[1,2])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="domain"):
        load_protocol(path, tmp_path)


def test_general_algorithm_cannot_be_marked_inapplicable_for_wide_input(tmp_path):
    path = saved_protocol(tmp_path)
    payload = json.loads(path.read_text())
    payload["thresholds"][0]["status"] = "not_applicable"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not run"):
        load_protocol(path, tmp_path)


@pytest.mark.parametrize("mutation", ["missing_attempt", "changed_median", "nonfinite_metric"])
def test_incomplete_or_changed_measurements_are_rejected(tmp_path, mutation):
    path = saved_protocol(tmp_path)
    payload = json.loads(path.read_text())
    record = payload["comparison"][0]["jacobi"]
    if mutation == "missing_attempt":
        record["attempts"] = []
    elif mutation == "changed_median":
        record["median_ms"] *= 2
    else:
        record["metrics"]["reconstruction_error"] = float("nan")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="timing|diagnostics"):
        load_protocol(path, tmp_path)


def test_failed_comparison_is_visible_and_not_averaged_with_successes(tmp_path):
    path = saved_protocol(tmp_path)
    payload = json.loads(path.read_text())
    record = payload["comparison"][0]["jacobi"]
    record.update(status="failed", stage="iteration", reason="Sweep limit", median_ms=None)
    record["attempts"][0].update(status="failed", stage="iteration", reason="Sweep limit")
    record["time_seconds"] = []
    path.write_text(json.dumps(payload))
    tables = render_tables(load_protocol(path, tmp_path))
    assert "Прототип Якоби & отказ" in tables


def test_failure_duration_and_stage_are_recorded(monkeypatch):
    import checks.contest_experiments as module
    from hsvd import NumericalBreakdownError

    def failed(*args, **kwargs):
        raise NumericalBreakdownError("Deliberate test failure", stage="pivot")

    monkeypatch.setattr(module, "jacobi_hsvd", failed)
    record = run_method(np.eye(2), 1, "jacobi")
    assert record["status"] == "failed" and record["stage"] == "pivot"
    assert record["elapsed_seconds"] >= 0
    json.dumps(record, allow_nan=False)


def test_missing_duration_of_a_failed_attempt_is_rejected(tmp_path):
    path = saved_protocol(tmp_path)
    payload = json.loads(path.read_text())
    record = payload["comparison"][0]["jacobi"]
    record.update(status="failed", reason="Sweep limit", median_ms=None, time_seconds=[])
    record["attempts"][0].update(status="failed", reason="Sweep limit", elapsed_seconds=None)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duration"):
        load_protocol(path, tmp_path)


def test_all_timed_attempts_survive_an_intermittent_failure(monkeypatch):
    import checks.contest_experiments as module
    case = analytic_cases()[9]
    success = run_method(case.A, case.p, "general")
    failure = dict(status="failed", stage="iteration", reason="Test failure", elapsed_seconds=.002)
    from copy import deepcopy
    records = iter(deepcopy(x) for x in [success, success, failure, success])
    monkeypatch.setattr(module, "run_method", lambda *a, **kw: next(records))
    result = timed_case(case, "general", 3)
    assert [x["status"] for x in result["attempts"]] == ["ok", "failed", "ok"]
    assert len(result["time_seconds"]) == 2
    assert result["status"] == "failed" and result["median_ms"] is None
    assert result["failed_repetition"] == 1


def test_high_precision_reference_uses_stored_entries_not_nominal_values():
    import mpmath as mp
    stored = np.nextafter(1., 2.)
    reference = high_precision_reference(np.array([[stored]]), 1)
    with mp.workdps(80):
        assert mp.mpf(reference["hyperbolic_values"][0]) == mp.mpf(float(stored))
    assert reference_relative_error([stored], reference) == 0
    assert reference_relative_error([1.], reference) > 2e-16


@pytest.mark.parametrize("complex_data", [False, True])
def test_high_precision_reference_forms_gram_before_rounding(complex_data):
    import mpmath as mp
    # The true (0,0) entry is -(2*d+d*d); binary64 Gram formation loses d*d.
    d = 2.**-30
    a = np.array([[1., 1.+d], [0., d]])
    if complex_data:
        a = a.astype(complex)
        a[:, 1] *= 1j
    reference = high_precision_reference(a, 1)
    with mp.workdps(80):
        delta = mp.mpf(d)
        trace = -2*delta-2*delta**2
        root = mp.sqrt(trace**2 + 4*delta**2)
        expected = [mp.sqrt((trace+root)/2), mp.sqrt((root-trace)/2)]
        for value, exact in zip(reference["hyperbolic_values"], expected):
            assert abs(mp.mpf(value)-exact)/exact < mp.mpf('1e-65')
