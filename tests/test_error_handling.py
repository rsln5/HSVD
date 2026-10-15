"""Input and failure contracts at the public factorization boundary."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

import hsvd.core as core
from hsvd import NumericalBreakdownError, hsvd
from hsvd._numeric import hermitian_part


def test_ragged_matrix_has_a_clear_error():
    with pytest.raises(ValueError, match="rectangular two-dimensional matrix"):
        hsvd([[1, 2], [3]], 1)


@pytest.mark.parametrize("name", ["rank_rtol", "spectral_rtol", "check_tol"])
@pytest.mark.parametrize("value", [10**1000, -(10**1000)])
def test_unbounded_integer_tolerance_has_a_clear_error(name, value):
    with pytest.raises(ValueError, match=name):
        hsvd(np.eye(2), 1, **{name: value})


def test_real_fraction_tolerance_is_supported():
    result = hsvd(np.eye(2), 1, check_tol=Fraction(1, 10**9))
    np.testing.assert_allclose(result.reconstruct(), np.eye(2))


def test_check_tolerance_cannot_underflow_to_zero():
    with pytest.raises(ValueError, match="positive in double precision"):
        hsvd(np.eye(2), 1, check_tol=Fraction(1, 10**1000))


@pytest.mark.parametrize("name", ["rank_rtol", "spectral_rtol", "check_tol"])
def test_tolerance_cannot_round_up_to_excluded_endpoint(name):
    almost_one = Fraction(10**30 - 1, 10**30)
    assert almost_one < 1 and float(almost_one) == 1.0
    with pytest.raises(ValueError, match=f"{name} must remain less than 1"):
        hsvd(np.eye(2), 1, **{name: almost_one})


@pytest.mark.parametrize("complex_data", [False, True])
def test_hermitian_part_preserves_extreme_finite_hermitian_matrix(complex_data):
    largest = np.finfo(float).max
    smallest = np.nextafter(0.0, 1.0)
    off_diagonal = complex(smallest, smallest) if complex_data else smallest
    mixed_components = complex(largest, smallest) if complex_data else largest
    matrix = np.array([[largest, off_diagonal, mixed_components],
                       [np.conj(off_diagonal), -largest, 0],
                       [np.conj(mixed_components), 0, 1]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = hermitian_part(matrix)
    np.testing.assert_array_equal(actual, matrix)


@pytest.mark.parametrize("failure", [np.linalg.LinAlgError, FloatingPointError])
def test_numeric_failure_preserves_cause_input_and_error_settings(monkeypatch, failure):
    matrix = np.array([[2.0, 1.0], [0.0, 1.0]])
    original = matrix.copy()
    cause = failure("simulated numerical failure")

    def fail_svd(*args, **kwargs):
        raise cause

    monkeypatch.setattr(np.linalg, "svd", fail_svd)
    with np.errstate(over="ignore", invalid="warn", divide="ignore"):
        settings = np.geterr()
        with pytest.raises(NumericalBreakdownError) as error:
            hsvd(matrix, 1)
        assert np.geterr() == settings
    assert error.value.stage == "arithmetic"
    assert error.value.__cause__ is cause
    assert error.value.diagnostics is None
    np.testing.assert_array_equal(matrix, original)


@pytest.mark.parametrize("field", [
    "reconstruction_error", "unitarity_error", "j_unitarity_error",
    "transform_error", "norm_v", "rank_truncation_error",
])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_diagnostics_are_rejected(monkeypatch, field, value):
    calculate = core._calculate_diagnostics

    def corrupted_diagnostics(*args, **kwargs):
        return replace(calculate(*args, **kwargs), **{field: value})

    monkeypatch.setattr(core, "_calculate_diagnostics", corrupted_diagnostics)
    with pytest.raises(NumericalBreakdownError, match="nonfinite") as error:
        hsvd(np.eye(2), 1)
    assert error.value.stage == "verification"
    assert not np.isfinite(getattr(error.value.diagnostics, field))
