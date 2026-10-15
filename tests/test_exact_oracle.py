"""Exact rational/complex arithmetic independently verifies a dense fixture."""
import numpy as np
import sympy as sp

from hsvd import hsvd


def test_dense_complex_example_against_exact_symbolic_oracle():
    # r=4, l+=1, l-=1, j=2; n=7 leaves one positive completion column.
    m, n, p = 5, 7, 4
    jmat = sp.diag(1, 1, 1, 1, -1, -1, -1)
    sigma = sp.zeros(m, n)
    sigma[0, 0], sigma[1, 4] = 3, 2
    sigma[2, 1] = sigma[2, 5] = 1
    sigma[3, 2] = sigma[3, 6] = 1
    v = sp.eye(n)
    for a, b, phase in [(0, 5, sp.I), (1, 4, 1), (2, 6, sp.I), (3, 5, 1)]:
        g = sp.eye(n)
        g[a, a] = g[b, b] = sp.Rational(5, 3)
        g[a, b] = sp.Rational(4, 3) * phase
        g[b, a] = sp.Rational(4, 3) * sp.conjugate(phase)
        v = v * g
    u = sp.eye(m)
    for a, b in [(0, 2), (1, 3), (2, 4)]:
        g = sp.eye(m)
        g[a, a] = g[b, b] = sp.Rational(3, 5)
        g[a, b] = sp.Rational(4, 5) * sp.I
        g[b, a] = sp.Rational(4, 5) * sp.I
        u = u * g
    assert v.H * jmat * v == jmat
    assert u.H * u == sp.eye(m)
    a_exact = u * sigma * v.H
    h_exact = a_exact * jmat * a_exact.H
    assert a_exact.rank() == 4
    assert h_exact.rank() == 2
    assert h_exact.eigenvals() == {sp.Integer(9): 1, sp.Integer(-4): 1, sp.Integer(0): 3}
    b_exact = a_exact.H * a_exact * jmat
    assert len((b_exact**2).nullspace()) - len(b_exact.nullspace()) == 2
    a = np.array(a_exact.evalf(17).tolist(), dtype=np.complex128)
    result = hsvd(a, p)
    assert (result.structure.rank, result.structure.l_plus,
            result.structure.l_minus, result.structure.j) == (4, 1, 1, 2)
    np.testing.assert_allclose(result.Sigma[0, 0], 3., rtol=1e-12)
    np.testing.assert_allclose(result.Sigma[1, 4], 2., rtol=1e-12)
    np.testing.assert_allclose(result.reconstruct(), a, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(result.V.conj().T @ result.J @ result.V,
                               result.J, atol=2e-11, rtol=2e-11)
