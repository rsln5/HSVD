"""Deterministic numerical checks of the paper's error estimates.

Run: python3 -m checks.verify_error_bounds
Optional: --output path/to/error_bounds.json

Only NumPy is required. The HSVD implementation is not imported: these
checks evaluate the stated algebraic maps, perturbations, derivatives,
and correction identities independently. They are floating-point checks
on specified inputs, not proofs, interval certificates, or benchmarks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261015


def adjoint(a):
    return a.conj().T


def norm(a):
    return float(np.linalg.norm(a, 2)) if a.ndim == 2 else float(np.linalg.norm(a))


def assert_bound(actual, bound):
    """Allow a small explicit rounding slack in these double-precision checks."""
    assert np.isfinite(actual) and np.isfinite(bound) and bound >= 0
    if actual > bound * (1 + 5e-10) + 5e-13:
        raise AssertionError(f"Measured {actual:.17g} exceeds bound {bound:.17g}")
    return actual / bound if bound else 0.


def orthonormal_columns(a):
    q, r = np.linalg.qr(a)
    diagonal = np.diag(r)
    return q * (diagonal / np.abs(diagonal))


def random_matrix(rng, shape, complex_data):
    a = rng.standard_normal(shape)
    return a + 1j*rng.standard_normal(shape) if complex_data else a


def neutral_blocks(v, d, z, signature):
    e = v @ d @ z
    f = signature @ v @ np.linalg.solve(d, z)
    g = f - e @ (adjoint(f) @ signature @ f) / 2
    return e, f, g


def verify_scaling():
    # An explicit isometry with two neutral directions and both nonzero signs.
    signature = np.diag([1., 1., 1., 1., -1., -1., -1., -1.])
    v = np.zeros((8, 4), dtype=complex)
    v[0, 0] = v[4, 1] = 1
    v[1, 2] = v[5, 2] = v[2, 3] = v[6, 3] = 1/np.sqrt(2)
    for i, j, c, s in ((0, 2, 3/5, 4/5), (1, 3, 5/13, 12/13)):
        rotation = np.eye(4, dtype=complex)
        rotation[np.ix_([i, j], [i, j])] = [[c, 1j*s], [1j*s, c]]
        v = v @ rotation
    d = np.diag([7., 5., 3., 2.])
    compressed = d @ adjoint(v) @ signature @ v @ d
    eigenvalues, eigenvectors = np.linalg.eigh(compressed)
    neutral_indices = np.flatnonzero(np.abs(eigenvalues) < 1e-10)
    assert len(neutral_indices) == 2
    z = eigenvectors[:, neutral_indices]
    z = z @ np.array([[3/5, 4j/5], [4j/5, 3/5]])
    e, f, g = neutral_blocks(v, d, z, signature)
    operator = v @ d @ d @ adjoint(v) @ signature
    residuals = (
        norm(adjoint(v) @ v - np.eye(4)),
        norm(adjoint(e) @ signature @ e),
        norm(adjoint(e) @ signature @ f - np.eye(2)),
        norm(adjoint(e) @ signature @ g - np.eye(2)),
        norm(adjoint(g) @ signature @ g),
        norm(operator @ e), norm(operator @ f-e), norm(operator @ g-e),
    )
    assert max(residuals) < 1e-10
    nonzero = np.flatnonzero(np.abs(eigenvalues) > 1e-10)
    x = (v @ d @ eigenvectors[:, nonzero]) / np.sqrt(np.abs(eigenvalues[nonzero]))
    t = np.column_stack((x, e, g))
    balanced = t / np.linalg.norm(t, axis=0)
    rows = []
    for scale in (.2, 2., 10.):
        es, fs, gs = neutral_blocks(v, scale*d, z, signature)
        assert_allclose(es, scale*e, atol=1e-11)
        assert_allclose(fs, f/scale, atol=1e-11)
        assert_allclose(gs, g/scale, atol=1e-11)
        xs = (v @ (scale*d) @ eigenvectors[:, nonzero]) / np.sqrt(np.abs(scale**2*eigenvalues[nonzero]))
        assert_allclose(xs, x, atol=1e-11)
        ts = np.column_stack((xs, es, gs))
        assert_allclose(ts / np.linalg.norm(ts, axis=0), balanced, atol=1e-11)
        pair = np.column_stack((es/2+gs, es/2-gs))
        expected_gram = np.diag([1., 1., -1., -1.])
        gram_error = norm(adjoint(pair) @ signature @ pair - expected_gram)
        assert gram_error < 2e-10
        predicted = scale**2*np.linalg.norm(e, 'fro')**2/2 + 2*np.linalg.norm(g, 'fro')**2/scale**2
        measured = float(np.linalg.norm(pair, 'fro')**2)
        assert_allclose(measured, predicted, rtol=1e-12, atol=1e-12)
        rows.append(dict(scale=scale, neutral_gram_error=gram_error,
                         squared_frobenius_norm=measured,
                         scale_identity_relative_error=abs(measured-predicted)/predicted))
    return dict(chains=2, max_chain_identity_residual=max(residuals), cases=rows)


def verify_block_bounds():
    rng = np.random.default_rng(SEED)
    signature = np.diag([1., 1., 1., 1., -1., -1., -1.])
    rows = []
    for complex_data in (False, True):
        for smallest in (.2, .02):
            v = orthonormal_columns(random_matrix(rng, (7, 4), complex_data))
            z = orthonormal_columns(random_matrix(rng, (4, 2), complex_data))
            vh = orthonormal_columns(v + 1e-5*random_matrix(rng, v.shape, complex_data))
            zh = orthonormal_columns(z + 1e-5*random_matrix(rng, z.shape, complex_data))
            d = np.diag([2., 1., .4, smallest])
            dh = d + np.diag(1e-5*smallest*rng.standard_normal(4))
            e, f, g = neutral_blocks(v, d, z, signature)
            eh, fh, gh = neutral_blocks(vh, dh, zh, signature)
            a = norm(d)
            dv, dd, dz = norm(vh-v), norm(dh-d), norm(zh-z)
            sigma_minus, a_plus = smallest-dd, a+dd
            assert sigma_minus > 0
            e_bound = a_plus*dv+dd+a*dz
            f_bound = dv/sigma_minus+dd/(smallest*sigma_minus)+dz/smallest
            g_bound = f_bound+e_bound/(2*sigma_minus**2)+a*(1/smallest+1/sigma_minus)*f_bound/2
            error = dict(E=norm(eh-e), F=norm(fh-f), G=norm(gh-g))
            bounds = dict(E=e_bound, F=f_bound, G=g_bound)
            lam = .03 if smallest == .2 else .003
            lam_hat = lam*(1+1e-5)
            direction, direction_hat = z[:, 0], zh[:, 0]
            gamma = min(lam, lam_hat)
            error['X'] = norm(vh @ dh @ direction_hat/np.sqrt(lam_hat)
                              - v @ d @ direction/np.sqrt(lam))
            bounds['X'] = ((a_plus*dv+dd+a*norm(direction_hat-direction))/np.sqrt(gamma)
                           + a*abs(lam_hat-lam)/(2*gamma**1.5))
            ratios = {name: assert_bound(error[name], bound) for name, bound in bounds.items()}
            rows.append(dict(complex_data=complex_data, smallest_retained_singular=smallest,
                             lambda_value=lam, measured=error, upper_bound=bounds, ratio=ratios))
    return dict(scope='algebraic maps; Z need not be a kernel for these inequalities', cases=rows)


def complement_at(theta):
    e = np.eye(4)
    t = np.column_stack((e[:, 0], np.cos(theta)*e[:, 1]+np.sin(theta)*e[:, 2]))
    # The plus sign is essential: this is the J-orthogonal complement.
    q = np.column_stack((np.sin(theta)*e[:, 1]+np.cos(theta)*e[:, 2], e[:, 3]))
    return t, q


def verify_complement():
    signature = np.diag([1., 1., -1., -1.])
    rows = []
    for theta in (.2, np.pi/4-.01, np.pi/4-.001):
        t, q = complement_at(theta)
        theta_hat = theta + .05*(np.pi/4-theta)
        th, qh = complement_at(theta_hat)
        assert_allclose(adjoint(t) @ signature @ q, np.zeros((2, 2)), atol=1e-14)
        assert_allclose(adjoint(th) @ signature @ qh, np.zeros((2, 2)), atol=1e-14)
        assert_allclose(adjoint(q) @ q, np.eye(2), atol=1e-14)
        sigma_t = float(np.linalg.svd(t, compute_uv=False)[-1])
        assert_allclose(sigma_t, 1., atol=1e-14)
        k, kh = adjoint(q) @ signature @ q, adjoint(qh) @ signature @ qh
        assert_allclose(k, np.diag([-np.cos(2*theta), -1.]), atol=1e-14)
        values, vectors = np.linalg.eigh(k)
        nu = float(np.min(np.abs(values)))
        v0 = (q @ vectors) / np.sqrt(np.abs(values))
        assert_allclose(norm(v0), 1/np.sqrt(nu), rtol=1e-12)
        assert_allclose(adjoint(v0) @ signature @ v0, -np.eye(2), atol=1e-11)
        epsilon_t = norm(th-t)
        projector_error = norm(qh @ adjoint(qh)-q @ adjoint(q))
        projector_ratio = assert_bound(projector_error, epsilon_t/sigma_t)
        # These explicit bases are already in the same continuous gauge.
        q_error = norm(qh-q)
        assert_bound(q_error, np.sqrt(2)*epsilon_t/sigma_t)
        epsilon_k = norm(kh-k)
        assert_bound(epsilon_k, 2*q_error)
        assert epsilon_k < nu
        vals_hat, vectors_hat = np.linalg.eigh(kh)
        assert np.array_equal(np.sign(vals_hat), np.sign(values))
        v0_hat = (qh @ vectors_hat) / np.sqrt(np.abs(vals_hat))
        assert_bound(norm(v0_hat), 1/np.sqrt(nu-epsilon_k))
        rows.append(dict(theta=theta, sigma_min_balanced=sigma_t, min_abs_eigenvalue=nu,
                         normalized_complement_norm=norm(v0), epsilon_T=epsilon_t,
                         projector_error=projector_error, projector_bound=epsilon_t/sigma_t,
                         projector_ratio=projector_ratio, epsilon_K=epsilon_k,
                         inertia_preserved=True))
    return dict(cases=rows)


def verify_derivative():
    rng = np.random.default_rng(SEED+1)
    signature = np.diag([1., -1., -1.])
    base = np.array([[2., .2, .4], [.1, 1.5, .7]])
    imaginary = np.array([[.2, -.3, .1], [.4, .2, -.2]])
    rows = []
    for complex_data in (False, True):
        a = base+1j*imaginary if complex_data else base
        values, vectors = np.linalg.eigh(a @ signature @ adjoint(a))
        for index, lam in enumerate(values):
            v = vectors[:, index]
            sigma = float(np.sqrt(abs(lam)))
            w = signature @ adjoint(a) @ v
            kappa_abs = norm(adjoint(a) @ v)/sigma
            maximizing = np.outer(v, w.conj())/norm(w)
            assert_allclose(norm(maximizing), 1., atol=1e-13)
            assert_allclose(np.linalg.norm(maximizing, 'fro'), 1., atol=1e-13)
            derivative = float(np.sign(lam)/sigma*np.real(adjoint(v) @ maximizing @ w))
            assert_allclose(abs(derivative), kappa_abs, rtol=1e-12)
            assert_allclose(kappa_abs, norm(adjoint(a) @ v/sigma), rtol=1e-12)
            for _ in range(3):
                direction = random_matrix(rng, a.shape, complex_data)
                direction /= norm(direction)
                directional = abs(float(np.real(adjoint(v) @ direction @ w))/sigma)
                assert_bound(directional, kappa_abs)
            differences = []
            for step in (1e-3, 1e-4, 1e-5):
                ap, am = a+step*maximizing, a-step*maximizing
                lp = np.linalg.eigvalsh(ap @ signature @ adjoint(ap))[index]
                lm = np.linalg.eigvalsh(am @ signature @ adjoint(am))[index]
                assert np.sign(lp) == np.sign(lm) == np.sign(lam)
                estimate = float((np.sqrt(abs(lp))-np.sqrt(abs(lm)))/(2*step))
                differences.append(dict(step=step, central_difference=estimate,
                                        relative_error=abs(estimate-derivative)/kappa_abs))
            assert differences[-1]['relative_error'] < 3e-8
            rows.append(dict(complex_data=complex_data, eigenvalue=float(lam), sigma=sigma,
                             kappa_abs=kappa_abs, kappa_rel=norm(a)*kappa_abs/sigma,
                             maximizing_direction_derivative=derivative,
                             finite_differences=differences))
    ordinary_values, ordinary_vectors = np.linalg.eigh(base @ base.T)
    for index, value in enumerate(ordinary_values):
        assert_allclose(norm(base.T @ ordinary_vectors[:, index])/np.sqrt(value), 1., atol=1e-12)
    return dict(ordinary_positive_signature_kappa_abs=1., cases=rows)


def inverse_sqrt_series(e, tolerance=1e-17):
    """Binomial series, valid for ||E||<1 even when E is nonnormal."""
    delta = norm(e)
    if not delta < 1:
        raise ValueError('The inverse-square-root series requires ||E||<1')
    result = np.eye(e.shape[0], dtype=e.dtype)
    power = result.copy()
    coefficient = 1.
    for k in range(1, 200):
        power = power @ e
        coefficient *= -(2*k-1)/(2*k)
        result += coefficient*power
        next_coefficient = abs(coefficient)*(2*k+1)/(2*k+2)
        # Formal exact-arithmetic tail majorant; it excludes rounding errors.
        tail = next_coefficient*delta**(k+1)/(1-delta)
        if tail < tolerance:
            return result, k, tail
    raise ArithmeticError('Series tail did not reach the requested tolerance')


def a_correction(delta):
    root = np.sqrt(1-delta)
    return delta/(root*(1+root))


def b_correction(delta):
    return delta/(1+np.sqrt(1-delta))


def verify_backward_correction():
    rng = np.random.default_rng(SEED+2)
    signature = np.diag([1., 1., -1., -1.])
    sigma = np.array([[2., 0., 0., 0.], [0., 1., 1., 0.], [0., 0., 0., 0.]])
    rows = []
    for complex_data in (False, True):
        u = orthonormal_columns(random_matrix(rng, (3, 3), complex_data))
        v = np.eye(4)
        for i, j, angle in ((0, 2, .4), (1, 3, -.3)):
            boost = np.eye(4)
            boost[np.ix_([i, j], [i, j])] = [[np.cosh(angle), np.sinh(angle)],
                                           [np.sinh(angle), np.cosh(angle)]]
            v = v @ boost
        if complex_data:
            v = v @ np.diag(np.exp(1j*np.array([.2, -.3, .4, -.1])))
        assert_allclose(adjoint(v) @ signature @ v, signature, atol=1e-13)
        a = u @ sigma @ adjoint(v)
        du_input = random_matrix(rng, u.shape, complex_data)
        dv_input = random_matrix(rng, v.shape, complex_data)
        uh = u+.02*du_input/norm(du_input)
        vh = v+.02*dv_input/norm(dv_input)
        du = adjoint(uh) @ uh-np.eye(3)
        dv = adjoint(vh) @ signature @ vh-signature
        delta_u, delta_v = norm(du), norm(dv)
        assert max(delta_u, delta_v) < 1
        vals_u, basis_u = np.linalg.eigh(adjoint(uh) @ uh)
        u_corrected = uh @ ((basis_u/np.sqrt(vals_u)) @ adjoint(basis_u))
        e = signature @ dv
        nonnormality = norm(adjoint(e) @ e-e @ adjoint(e))
        assert nonnormality > 1e-10
        correction, terms, tail = inverse_sqrt_series(e)
        v_corrected = vh @ correction
        orth_error = norm(adjoint(u_corrected) @ u_corrected-np.eye(3))
        j_orth_error = norm(adjoint(v_corrected) @ signature @ v_corrected-signature)
        assert max(orth_error, j_orth_error) < 1e-11
        u_bound, v_bound = b_correction(delta_u), norm(vh)*a_correction(delta_v)
        u_ratio = assert_bound(norm(u_corrected-uh), u_bound)
        v_ratio = assert_bound(norm(v_corrected-vh), v_bound)
        l = uh @ sigma
        ah = l @ adjoint(vh)
        residual = a-ah
        ac = u_corrected @ sigma @ adjoint(v_corrected)
        error = float(np.linalg.norm(a-ac, 'fro'))
        bound = float(np.linalg.norm(residual, 'fro')+np.linalg.norm(sigma, 'fro')*norm(vh)
                      *(b_correction(delta_u)+a_correction(delta_v)))
        backward_ratio = assert_bound(error, bound)
        # Relative bound using the Frobenius diagnostics, rather than exact spectral defects.
        upper_u, upper_v = float(np.linalg.norm(du, 'fro')), float(np.linalg.norm(dv, 'fro'))
        assert max(upper_u, upper_v) < 1
        norm_a = float(np.linalg.norm(a, 'fro'))
        rho = float(np.linalg.norm(sigma, 'fro')*norm(vh)/norm_a)
        relative_bound = (np.linalg.norm(residual, 'fro')/norm_a
                          + rho*(b_correction(upper_u)+a_correction(upper_v)))
        assert_bound(error/norm_a, relative_bound)
        # Independent identities linking the two residuals and the left defect.
        transformed = a @ signature @ vh @ signature-l
        left_defect = signature @ vh @ signature @ adjoint(vh)-np.eye(4)
        identities = (
            norm(transformed-residual @ signature @ vh @ signature-l @ dv @ signature),
            norm(residual-transformed @ adjoint(vh)+a @ left_defect),
            norm(left_defect-np.linalg.solve(adjoint(vh), dv @ signature @ adjoint(vh))),
        )
        assert max(identities) < 1e-11
        rows.append(dict(complex_data=complex_data, delta_U=delta_u, delta_V=delta_v,
                         nonnormality_JD_V=nonnormality, series_terms=terms,
                         formal_series_tail_bound=tail,
                         corrected_unitarity_error=orth_error, corrected_j_unitarity_error=j_orth_error,
                         U_correction_ratio=u_ratio, V_correction_ratio=v_ratio,
                         structured_backward_error=error, backward_upper_bound=bound,
                         backward_ratio=backward_ratio, relative_upper_bound=float(relative_bound),
                         max_residual_identity_error=max(identities)))
    return dict(canonical_structure=dict(rank=2, positive=1, negative=0, isotropic=1), cases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/audit/error_bounds.json')
    args = parser.parse_args()
    report = dict(protocol='error-bounds-v1', status='passed', seed=SEED,
                  environment=dict(python=platform.python_version(), numpy=np.__version__),
                  scope='deterministic floating-point consistency checks; no interval certification or benchmark',
                  bound_check_slack=dict(rtol=5e-10, atol=5e-13),
                  scaling=verify_scaling(), block_perturbations=verify_block_bounds(),
                  complement=verify_complement(), derivative=verify_derivative(),
                  backward_correction=verify_backward_correction(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    counts = {key: len(report[key]['cases']) for key in
              ('scaling', 'block_perturbations', 'complement', 'derivative', 'backward_correction')}
    print(json.dumps(dict(status=report['status'], cases=counts, output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
