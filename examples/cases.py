"""Independent test matrices with known HSVD structure.

The generator uses products of explicit plane rotations/boosts, not the
SVD/eigenvector construction used by the implementation.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Case:
    name: str
    A: np.ndarray
    p: int
    l_plus: int
    l_minus: int
    j: int
    positive_values: tuple = ()
    negative_values: tuple = ()


def analytic_cases():
    vr = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [.5, -.5], [.5, -.5]])
    return [
        Case("zero", np.zeros((3, 4)), 2, 0, 0, 0),
        Case("positive_row", np.array([[2., 1.]]), 1, 1, 0, 0, (np.sqrt(3),)),
        Case("negative_row", np.array([[1., 2.]]), 1, 0, 1, 0, (), (np.sqrt(3),)),
        Case("isotropic_row", np.array([[1., 1.]]), 1, 0, 0, 1),
        Case("complex_isotropic_row", np.array([[1., 1j]]), 1, 0, 0, 1),
        Case("two_isotropic_pairs", np.array([[2., 0., 2., 0.], [0., 1., 0., 1.]]), 2, 0, 0, 2),
        Case("rank_deficient_j0", np.diag([1., 0.]), 1, 1, 0, 0, (1.,)),
        Case("different_C_H_kernels", np.diag([2., 1.]) @ vr.T, 2, 1, 0, 1, (np.sqrt(2.5),)),
        Case("complex_column", np.array([[1.], [1j], [2.]]), 0, 0, 1, 0, (), (np.sqrt(6),)),
        Case("scalar_positive", np.array([[3. + 4j]]), 1, 1, 0, 0, (5.,)),
        Case("scalar_negative", np.array([[3. + 4j]]), 0, 0, 1, 0, (), (5.,)),
    ]


def generated_case(m, p, q, lp, lm, j, *, seed=0, complex_data=False,
                   rotation_size=.4, positive=None, negative=None):
    """Construct A from a known canonical Sigma and independent U,V.

    Each plane transformation is exactly J-unitary over exact arithmetic.
    Returned expected singular values are independent of the factorization.
    """
    n = p + q
    if min(p - lp - j, q - lm - j, m - lp - lm - j) < 0:
        raise ValueError("Invalid canonical dimensions")
    rng = np.random.default_rng(seed)
    dtype = complex if complex_data else float
    positive = tuple(np.linspace(2.5, 1., lp) if positive is None else positive)
    negative = tuple(np.linspace(2., .8, lm) if negative is None else negative)
    if len(positive) != lp or len(negative) != lm:
        raise ValueError("Incorrect number of canonical values")
    sigma = np.zeros((m, n))
    sigma[np.arange(lp), np.arange(lp)] = positive
    sigma[lp + np.arange(lm), p + np.arange(lm)] = negative
    sigma[lp + lm + np.arange(j), lp + np.arange(j)] = 1
    sigma[lp + lm + np.arange(j), p + lm + np.arange(j)] = 1
    x = rng.standard_normal((m, m)).astype(dtype)
    if complex_data:
        x += 1j * rng.standard_normal((m, m))
    u, _ = np.linalg.qr(x)
    v = np.eye(n, dtype=dtype)
    signs = np.r_[np.ones(p), -np.ones(q)]
    if n > 1:
        for _ in range(3 * n):
            a, b = sorted(rng.choice(n, size=2, replace=False))
            angle = rng.uniform(-rotation_size, rotation_size)
            phase = np.exp(1j * rng.uniform(-np.pi, np.pi)) if complex_data else 1.
            if signs[a] == signs[b]:
                c, s = np.cos(angle), np.sin(angle)
                block = np.array([[c, phase * s], [-np.conj(phase) * s, c]])
            else:
                c, s = np.cosh(angle), np.sinh(angle)
                block = np.array([[c, phase * s], [np.conj(phase) * s, c]])
            v[:, [a, b]] = v[:, [a, b]] @ block
    np.testing.assert_allclose(v.conj().T @ (signs[:, None] * v), np.diag(signs), atol=2e-12, rtol=2e-12)
    a = u @ sigma @ v.conj().T
    name = f"generated_{m}_{p}_{q}_{lp}_{lm}_{j}_{seed}_{'complex' if complex_data else 'real'}"
    return Case(name, a, p, lp, lm, j, positive, negative)


SPECS = [
    (8, 3, 2, 3, 2, 0),       # Tall, full column rank.
    (3, 4, 3, 1, 1, 1),       # Wide, with a null pair and a complement.
    (6, 5, 4, 2, 1, 2),       # Mixed spectrum, several coupled chains.
    (6, 4, 3, 0, 0, 3),       # Purely isotropic, with zero rows/columns.
    (4, 5, 0, 3, 0, 0),       # Positive definite signature (ordinary SVD).
    (4, 0, 5, 0, 3, 0),       # Negative definite signature.
    (5, 3, 3, 1, 1, 0),       # Rank deficient, j=0, both complements.
    (4, 2, 2, 1, 1, 1),       # Square, rank deficient, no complement.
    (2, 3, 3, 0, 0, 2),       # Wide, pure isotropic, both complements.
    (4, 3, 2, 0, 1, 1),       # Empty positive nonzero block.
    (4, 3, 2, 1, 0, 1),       # Empty negative nonzero block.
    (5, 4, 1, 2, 0, 0),       # One signature sector unused by A.
]


def standard_cases():
    yield from analytic_cases()
    for complex_data in (False, True):
        for seed in (17, 31, 2026, 20260911):
            for spec in SPECS:
                yield generated_case(*spec, seed=seed, complex_data=complex_data)
