import numpy as np
import pytest
from fractions import Fraction

from checks.contest_experiments import factor_metrics, jacobi_hsvd, jacobi_applicability, run_method
from examples.cases import generated_case


@pytest.mark.parametrize("complex_data",[False,True])
@pytest.mark.parametrize("p",[0,2,4])
def test_jacobi_recovers_full_factors_and_known_values(complex_data,p):
    case=generated_case(7,p,4-p,p,4-p,0,seed=12,complex_data=complex_data)
    u,s,v,metrics,_=jacobi_hsvd(case.A,p)
    j=np.diag(np.r_[np.ones(p),-np.ones(4-p)])
    np.testing.assert_allclose(u@s@v.conj().T,case.A,atol=2e-11)
    np.testing.assert_allclose(u.conj().T@u,np.eye(7),atol=2e-11)
    np.testing.assert_allclose(v.conj().T@j@v,j,atol=2e-11)
    np.testing.assert_allclose(np.diag(s),np.r_[case.positive_values,case.negative_values],atol=2e-11)


def test_equal_norm_orthogonal_pivot_is_rotated():
    a=np.array([[1.,.6],[0.,.8]])
    _,s,_,_,it=jacobi_hsvd(a,2)
    np.testing.assert_allclose(np.diag(s),np.linalg.svd(a,compute_uv=False),atol=2e-14)
    assert it["rotations"]>0


def test_complex_phase_and_hyperbolic_accumulator():
    a=np.array([[2.,1j],[1.,2j],[1j,0.]])
    u,s,v,_,_=jacobi_hsvd(a,1)
    np.testing.assert_allclose(u@s@v.conj().T,a,atol=2e-12)
    eigenvalues=np.linalg.eigvalsh(a@np.diag([1,-1])@a.conj().T)
    np.testing.assert_allclose(np.sort([s[0,0]**2,-s[1,1]**2]),eigenvalues[[0,-1]],atol=2e-12)


def test_jacobi_rejects_rank_deficiency_and_iteration_limit():
    with pytest.raises(ValueError):
        jacobi_hsvd(np.ones((2,2)),1)
    with pytest.raises(ArithmeticError):
        jacobi_hsvd(np.array([[2.,1.,0.],[1.,3.,1.],[0.,1.,2.]]),1,max_sweeps=1)


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1e200, 1e300])
@pytest.mark.parametrize("complex_data", [False, True])
def test_extreme_scale_keeps_finite_relative_diagnostics(scale, complex_data):
    a = np.array([[2., 1.], [0., 1.], [1., -1.]])
    if complex_data:
        a = a * (1 + .5j)
    u, s, v, metrics, _ = jacobi_hsvd(a * scale, 1)
    assert all(np.isfinite(value) for value in metrics.values())
    np.testing.assert_allclose(u @ (s / scale) @ v.conj().T, a, atol=2e-13)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(3), atol=2e-13)
    j = np.diag([1., -1.])
    np.testing.assert_allclose(v.conj().T @ j @ v, j, atol=2e-13)


@pytest.mark.parametrize("scale", [1e-300, 1., 1e300])
def test_relative_diagnostics_detect_known_one_percent_error_at_any_scale(scale):
    a = np.array([[scale]])
    metrics = factor_metrics(a, 1, np.eye(1), a * 1.01, np.eye(1))
    assert metrics["reconstruction_error"] == pytest.approx(.01)
    assert metrics["transform_error"] == pytest.approx(.01 / 1.01)


@pytest.mark.parametrize("value", [np.nan, np.inf, 0., -1., True])
def test_invalid_check_tolerance_cannot_bypass_verification(value):
    with pytest.raises(ValueError):
        jacobi_hsvd(np.eye(2), 1, check_tol=value)


def test_unrepresentable_singular_value_is_reported_as_failure():
    with pytest.raises(ArithmeticError):
        jacobi_hsvd(np.full((4, 1), 1e308), 1)


def test_nonfinite_diagnostics_are_never_accepted(monkeypatch):
    import json
    import checks.contest_experiments as module
    original = module.factor_metrics

    def nonfinite_metrics(*args):
        result = original(*args)
        result["reconstruction_error"] = np.nan
        return result

    monkeypatch.setattr(module, "factor_metrics", nonfinite_metrics)
    with pytest.raises(ArithmeticError):
        jacobi_hsvd(np.eye(2), 1)
    failed = module.run_method(np.eye(2), 1, "jacobi")
    assert failed["status"] == "failed"
    assert failed["metrics"]["reconstruction_error"] is None
    assert failed["nonfinite_metrics"] == ["reconstruction_error"]
    json.dumps(failed, allow_nan=False)


@pytest.mark.parametrize("name", ["tol", "check_tol"])
@pytest.mark.parametrize("value", [10**1000, -(10**1000), Fraction(1, 10**1000)])
def test_tolerance_conversion_has_a_clear_error(name, value):
    with pytest.raises(ValueError, match=name):
        jacobi_hsvd(np.eye(2), 1, **{name: value})


def test_real_fraction_tolerances_are_supported():
    _, _, _, metrics, _ = jacobi_hsvd(
        np.eye(2), 1, tol=Fraction(1, 10**13), check_tol=Fraction(1, 10**9))
    assert metrics["reconstruction_error"] == 0


def test_iteration_failure_has_a_diagnostic_stage():
    from hsvd import NumericalBreakdownError

    with pytest.raises(NumericalBreakdownError) as error:
        jacobi_hsvd(np.array([[2.,1.,0.],[1.,3.,1.],[0.,1.,2.]]), 1, max_sweeps=1)
    assert error.value.stage == "iteration"


@pytest.mark.parametrize("p", [1, 2])
def test_convergence_on_last_permitted_sweep_is_accepted(p):
    a = np.array([[2., 1.], [1., 3.]])
    u, sigma, v, _, iterations = jacobi_hsvd(a, p, max_sweeps=1)
    assert iterations["sweeps"] == 1
    np.testing.assert_allclose(u @ sigma @ v.T, a, atol=2e-14)


@pytest.mark.parametrize("name", ["tol", "check_tol"])
def test_tolerance_rounding_up_to_one_is_rejected(name):
    with pytest.raises(ValueError, match=name):
        jacobi_hsvd(np.eye(2), 1, **{name: Fraction(10**100 - 1, 10**100)})


@pytest.mark.parametrize("complex_data", [False, True])
def test_repeated_values_and_sign_reversal_preserve_the_correct_sectors(complex_data):
    case = generated_case(6, 2, 2, 2, 2, 0, seed=31, complex_data=complex_data,
                          positive=[2., 2.], negative=[.5, .5])
    # Swap the two signature sectors and negate J. Positive/negative values
    # must swap; arbitrary rotations within a repeated eigenspace are valid.
    permutation = [2, 3, 0, 1]
    a = case.A[:, permutation]
    u, sigma, v, _, _ = jacobi_hsvd(a, 2)
    np.testing.assert_allclose(np.diag(sigma), [.5, .5, 2., 2.], atol=2e-13)
    np.testing.assert_allclose(u @ sigma @ v.conj().T, a, atol=2e-13)


@pytest.mark.parametrize("phase", [1., 1j, np.exp(.37j)])
def test_nearly_parallel_opposite_signature_columns(phase):
    delta = 1e-3
    a = np.array([[1., phase], [0., delta*phase]], dtype=np.result_type(phase, float))
    u, sigma, v, metrics, _ = jacobi_hsvd(a, 1)
    # Analytic eigenvalues of [[0,-delta*conj(phase)*phase],[-delta,-delta²]]:
    # trace=-delta² and determinant=-delta², independent of the column phase.
    root = np.hypot(delta*delta, 2*delta)
    expected = np.sqrt([(root-delta*delta)/2, (root+delta*delta)/2])
    np.testing.assert_allclose(np.diag(sigma), expected, rtol=2e-10)
    np.testing.assert_allclose(u @ sigma @ v.conj().T, a, atol=2e-12)
    assert metrics["norm_v"] > 10  # exercises a genuine hyperbolic boost


@pytest.mark.parametrize("a", [np.ones((1, 2)), np.array([[1., 1.], [0., 0.]]),
                               np.zeros((3, 2))])
def test_outside_jacobi_domain_is_skipped_without_calling_the_prototype(a, monkeypatch):
    import checks.contest_experiments as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Jacobi must not be invoked on this input")

    monkeypatch.setattr(module, "jacobi_hsvd", forbidden)
    assert not jacobi_applicability(a)["applicable"]
    baseline = run_method(a, 1, "jacobi")
    assert baseline["status"] == "not_applicable"
    assert baseline["elapsed_seconds"] is None
    general = run_method(a, 1, "general")
    assert general["status"] == "ok"
    assert general["elapsed_seconds"] >= 0
