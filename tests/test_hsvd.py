"""Checks use independent matrix identities and known canonical invariants."""
import numpy as np
import pytest

from examples.cases import SPECS, analytic_cases, generated_case
from hsvd import NumericalBreakdownError, hsvd


def check_result(a, p, result, *, atol=2e-10, expected=None):
    m, n = a.shape
    u, sigma, v = result.U, result.Sigma, result.V
    jmat = np.diag(np.r_[np.ones(p), -np.ones(n - p)])
    s = result.structure
    assert (u.shape, sigma.shape, v.shape) == ((m, m), (m, n), (n, n))
    assert sigma.dtype.kind == "f"
    assert (u.dtype.kind == "c") == np.iscomplexobj(a)
    assert (v.dtype.kind == "c") == np.iscomplexobj(a)
    assert np.isfinite(u).all() and np.isfinite(v).all() and np.isfinite(sigma).all()
    assert s.rank == s.l_plus + s.l_minus + s.j
    assert s.p0 == p - s.l_plus - s.j >= 0
    assert s.q0 == n - p - s.l_minus - s.j >= 0
    assert s.rank <= min(m, n)
    np.testing.assert_allclose(result.J, jmat, atol=0, rtol=0)
    # Check the returned factors directly, independently of stored diagnostics.
    scale = max(float(np.max(np.abs(a))), np.finfo(float).tiny)
    np.testing.assert_allclose((u @ (sigma / scale)) @ v.conj().T,
                               a / scale, atol=atol, rtol=atol)
    np.testing.assert_allclose(np.einsum("ki,kj->ij", u.conj(), u, optimize=False),
                               np.eye(m), atol=atol, rtol=0)
    np.testing.assert_allclose(v.conj().T @ jmat @ v, jmat, atol=atol, rtol=0)
    np.testing.assert_allclose(a @ jmat @ v @ jmat, u @ sigma,
                               atol=atol * scale, rtol=atol)
    np.testing.assert_allclose(result.reconstruct() / scale, a / scale, atol=atol, rtol=atol)
    # The sparse canonical pattern, including *unit* isotropic blocks.
    mask = np.zeros((m, n), dtype=bool)
    pos = np.arange(s.l_plus)
    neg = np.arange(s.l_minus)
    iso = np.arange(s.j)
    mask[pos, pos] = True
    mask[s.l_plus + neg, p + neg] = True
    mask[s.l + iso, s.l_plus + iso] = True
    mask[s.l + iso, p + s.l_minus + iso] = True
    assert np.count_nonzero(sigma[~mask]) == 0
    np.testing.assert_array_equal(sigma[s.l + iso, s.l_plus + iso], np.ones(s.j))
    np.testing.assert_array_equal(sigma[s.l + iso, p + s.l_minus + iso], np.ones(s.j))
    positive = sigma[pos, pos]
    negative = sigma[s.l_plus + neg, p + neg]
    assert np.all(positive > 0) and np.all(negative > 0)
    assert np.all(np.diff(positive) <= 0) and np.all(np.diff(negative) <= 0)
    if expected is not None:
        assert (s.l_plus, s.l_minus, s.j) == (expected.l_plus, expected.l_minus, expected.j)
        np.testing.assert_allclose(positive, sorted(expected.positive_values, reverse=True), atol=atol, rtol=atol)
        np.testing.assert_allclose(negative, sorted(expected.negative_values, reverse=True), atol=atol, rtol=atol)
    # The nilpotent block coefficient is 1, not 1/2. Form B only in tests.
    if s.j:
        b = a.conj().T @ a @ jmat
        vv = v[:, s.l_plus:s.l_plus + s.j]
        vw = v[:, p + s.l_minus:p + s.l_minus + s.j]
        np.testing.assert_allclose(b @ vv, vv + vw, atol=atol, rtol=atol)
        np.testing.assert_allclose(b @ vw, -(vv + vw), atol=atol, rtol=atol)
    # Columns added by completion really belong to ker(AJ).
    v0 = np.column_stack((v[:, s.l_plus + s.j:p], v[:, p + s.l_minus + s.j:]))
    np.testing.assert_allclose(a @ jmat @ v0, 0, atol=atol * scale)


@pytest.mark.parametrize("case", analytic_cases(), ids=lambda c: c.name)
def test_analytic_examples(case):
    check_result(case.A, case.p, hsvd(case.A, case.p), expected=case)


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("seed", [17, 31, 2026, 20260911])
@pytest.mark.parametrize("spec", SPECS)
def test_known_structure_under_independent_transformations(spec, seed, complex_data):
    case = generated_case(*spec, seed=seed, complex_data=complex_data)
    check_result(case.A, case.p, hsvd(case.A, case.p), expected=case)


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("shape,p", [((1, 5), 2), ((3, 7), 3), ((8, 3), 1), ((5, 5), 4)])
def test_generic_random_matrices_against_direct_hermitian_spectrum(shape, p, complex_data):
    rng = np.random.default_rng(4242)
    a = rng.standard_normal(shape)
    if complex_data:
        a = a + 1j * rng.standard_normal(shape)
    result = hsvd(a, p)
    check_result(a, p, result)
    assert result.structure.rank == min(shape) and result.structure.j == 0
    eigenvalues = np.linalg.eigvalsh(a @ result.J @ a.conj().T)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-10]
    computed = np.r_[np.diag(result.Sigma[:result.structure.l_plus, :result.structure.l_plus])**2,
                     -np.diag(result.Sigma[result.structure.l_plus:result.structure.l,
                                           p:p + result.structure.l_minus])**2]
    np.testing.assert_allclose(np.sort(computed), eigenvalues, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("complex_data", [False, True])
def test_repeated_nonzero_eigenvalues_and_multiple_zero_chains(complex_data):
    case = generated_case(6, 5, 4, 2, 2, 2, seed=21, complex_data=complex_data,
                          positive=[2., 2.], negative=[1., 1.])
    check_result(case.A, case.p, hsvd(case.A, case.p), expected=case)


def test_multiple_chains_really_exercise_off_diagonal_gram_correction():
    case = generated_case(6, 5, 4, 2, 1, 2, seed=17, complex_data=True)
    # Audit the fixture: an implementation that corrects only diag(F*JF)
    # must not be able to pass this example accidentally.
    ur, singulars, vh = np.linalg.svd(case.A, full_matrices=False)
    r = case.l_plus + case.l_minus + case.j
    s, vr = singulars[:r], vh.conj().T[:, :r]
    jmat = np.diag(np.r_[np.ones(case.p), -np.ones(case.A.shape[1] - case.p)])
    c = s[:, None] * (vr.conj().T @ jmat @ vr) * s
    lam, z = np.linalg.eigh(c)
    z0 = z[:, np.abs(lam) < 1e-10]
    f = jmat @ vr @ (z0 / s[:, None])
    d = f.conj().T @ jmat @ f
    assert np.linalg.norm(d - np.diag(np.diag(d))) > 1e-3
    result = hsvd(case.A, case.p)
    check_result(case.A, case.p, result, expected=case)


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1e-12, 1., 1e12, 1e200, 1e300])
@pytest.mark.parametrize("complex_data", [False, True])
def test_extreme_uniform_scale_without_isotropic_blocks(scale, complex_data):
    a = np.array([[2., 1.], [0., 1.], [1., -1.]])
    if complex_data:
        a = a * (1 + .5j)
    reference = hsvd(a, 1)
    result = hsvd(a * scale, 1)
    check_result(a * scale, 1, result)
    assert result.structure == reference.structure
    np.testing.assert_allclose(result.Sigma / scale, reference.Sigma, atol=3e-14, rtol=3e-14)


@pytest.mark.parametrize("scale", [.05, .5, 2., 20.])
@pytest.mark.parametrize("complex_data", [False, True])
def test_scaling_preserves_unit_blocks_in_sigma(scale, complex_data):
    a = np.array([[1., 1j]]) if complex_data else np.array([[1., 1.]])
    result = hsvd(scale * a, 1)
    check_result(scale * a, 1, result)
    assert result.structure.j == 1
    np.testing.assert_array_equal(result.Sigma, [[1., 1.]])


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_unresolvable_canonical_isotropic_scale_is_not_reported_as_success(scale):
    with pytest.raises(NumericalBreakdownError):
        hsvd(scale * np.array([[1., 1.]]), 1)


def test_too_large_spectral_cutoff_cannot_invent_impossible_structure():
    a = np.diag([1., 1e-8])
    with pytest.raises(NumericalBreakdownError, match="incompatible") as error:
        hsvd(a, 1)
    assert error.value.stage == "spectrum"
    result = hsvd(a, 1, spectral_rtol=1e-18)
    check_result(a, 1, result)
    assert result.structure.j == 0 and result.structure.rank == 2


def test_rank_truncation_is_measured_against_original_input():
    a = np.diag([1., 1e-12])
    result = hsvd(a, 1, rank_rtol=1e-10)
    assert result.structure.rank == 1 and result.structure.j == 0
    assert result.diagnostics.rank_truncation_error == pytest.approx(1e-12)
    assert result.diagnostics.reconstruction_error == pytest.approx(1e-12)
    with pytest.raises(NumericalBreakdownError) as error:
        hsvd(a, 1, rank_rtol=1e-10, check_tol=1e-13)
    assert error.value.stage == "verification"
    assert error.value.diagnostics.reconstruction_error > 1e-13


@pytest.mark.parametrize("tail", [1e-200, 1e-300])
def test_tiny_discarded_singular_value_is_not_reported_as_zero(tail):
    result = hsvd(np.diag([1., tail]), 1)
    assert result.structure.rank == 1
    assert result.diagnostics.rank_truncation_error == pytest.approx(tail, rel=1e-12, abs=0)
    assert result.diagnostics.reconstruction_error == pytest.approx(tail, rel=1e-12, abs=0)


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("p", [0, 2])
def test_diagonal_subnormal_gram_value_is_not_erased_by_symmetrization(p, complex_data):
    # Squaring this exactly representable power of two gives the smallest
    # subnormal. The Gram matrix and both singular values are representable.
    tiny_value = np.sqrt(np.nextafter(0.0, 1.0))
    a = np.diag([1.0, tiny_value]).astype(complex if complex_data else float)
    result = hsvd(a, p, rank_rtol=0, spectral_rtol=0)
    assert result.structure.rank == 2 and result.structure.j == 0
    np.testing.assert_allclose(np.diag(result.Sigma) / [1.0, tiny_value], 1.0,
                               rtol=2e-14, atol=0)
    np.testing.assert_allclose(result.V.conj().T @ result.J @ result.V,
                               result.J, rtol=0, atol=2e-14)
    np.testing.assert_allclose(result.reconstruct()[:, 1] / tiny_value,
                               a[:, 1] / tiny_value, rtol=2e-14, atol=0)


@pytest.mark.parametrize("complex_data", [False, True])
def test_subnormal_off_diagonal_gram_entries_preserve_indefinite_spectrum(complex_data):
    tiny_value = np.sqrt(np.nextafter(0.0, 1.0))
    rotation = np.array([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2)
    a = np.zeros((3, 3), dtype=complex if complex_data else float)
    a[0, 0] = 1
    a[1:, 1:] = np.diag([tiny_value, 3 * tiny_value]) @ rotation.T
    result = hsvd(a, 2, rank_rtol=0, spectral_rtol=0)
    assert (result.structure.l_plus, result.structure.l_minus, result.structure.j) == (2, 1, 0)
    # The bottom Gram block has eigenvalues +/-3*tiny_value**2.
    values = result.Sigma[[1, 2], [1, 2]] / (np.sqrt(3) * tiny_value)
    np.testing.assert_allclose(values, 1, rtol=2e-14, atol=0)
    np.testing.assert_allclose(result.V.conj().T @ result.J @ result.V,
                               result.J, rtol=0, atol=2e-14)


@pytest.mark.parametrize("shape,p", [((2, 5), 2), ((6, 2), 0)])
@pytest.mark.parametrize("complex_data", [False, True])
def test_exact_zero_factors_and_diagnostics(shape, p, complex_data):
    a = np.zeros(shape, dtype=complex if complex_data else float)
    result = hsvd(a, p)
    check_result(a, p, result)
    assert result.input_scale == 0
    assert result.diagnostics.norm_v == 1
    assert result.diagnostics.rank_truncation_error == 0
    np.testing.assert_array_equal(result.singular_values_scaled, np.zeros(min(shape)))


def test_plausible_but_wrong_zero_classification_fails_final_checks():
    a = np.array([[1., 1. + 1e-7]])
    with pytest.raises(NumericalBreakdownError) as error:
        hsvd(a, 1, spectral_rtol=1e-4, check_tol=1e-10)
    assert error.value.stage == "verification"
    assert error.value.diagnostics is not None
    accurate = hsvd(a, 1, check_tol=1e-8)
    assert accurate.structure.j == 0


@pytest.mark.parametrize("delta,expected", [(1e-3, (0, 1, 0)), (-1e-3, (1, 0, 0)), (0., (0, 0, 1))])
def test_change_of_inertia_across_an_isotropic_row(delta, expected):
    result = hsvd(np.array([[1., 1. + delta]]), 1)
    s = result.structure
    assert (s.l_plus, s.l_minus, s.j) == expected


def test_no_mutation_and_real_double_precision_promotion():
    a = np.array([[2, 1], [0, 1]], dtype=np.int64)
    before = a.copy()
    result = hsvd(a, np.int64(1))
    np.testing.assert_array_equal(a, before)
    assert result.U.dtype == np.float64 and result.V.dtype == np.float64
    b = a.astype(np.float32)
    assert hsvd(b, 1).U.dtype == np.float64
    c = a.astype(np.complex64)
    assert hsvd(c, 1).U.dtype == np.complex128


@pytest.mark.parametrize("a", [[], [1, 2], np.zeros((0, 3)), np.zeros((3, 0)),
                              np.zeros((2, 2, 2)), [[np.nan]], [[np.inf]], [[1 + np.inf * 1j]]])
def test_invalid_matrix(a):
    with pytest.raises(ValueError):
        hsvd(a, 0)


@pytest.mark.parametrize("a", [[["1"]], [[None]], np.array([[1]], dtype=object)])
def test_nonnumeric_matrix(a):
    with pytest.raises(TypeError):
        hsvd(a, 0)


@pytest.mark.parametrize("p", [-1, 3, .5, 1., True, "1", None])
def test_invalid_signature(p):
    with pytest.raises(ValueError):
        hsvd(np.eye(2), p)


@pytest.mark.parametrize("name", ["rank_rtol", "spectral_rtol", "check_tol"])
@pytest.mark.parametrize("value", [-1., 1., np.nan, np.inf, True, "1e-8", 1j])
def test_invalid_tolerance(name, value):
    with pytest.raises(ValueError):
        hsvd(np.eye(2), 1, **{name: value})


def test_check_cannot_be_disabled():
    with pytest.raises(ValueError):
        hsvd(np.eye(2), 1, check_tol=0)


def test_non_contiguous_input_and_readonly_array():
    a = np.arange(30., dtype=float).reshape(5, 6)[::2, ::2]
    a.flags.writeable = False
    result = hsvd(a, 2)
    check_result(a, 2, result)
