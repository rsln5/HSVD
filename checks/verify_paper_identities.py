"""Independent exact-arithmetic checks of identities printed in the paper.

This does not call either HSVD implementation and is not a substitute for
the general proofs. Run: python3 -m checks.verify_paper_identities
"""
import json
from pathlib import Path

import sympy as sp


def verify():
    count = 0
    for m in range(1, 5):
        for n in range(1, 6):
            for p in range(n + 1):
                q = n - p
                signature = sp.diag(*([1] * p + [-1] * q))
                for lp in range(p + 1):
                    for lm in range(q + 1):
                        for j in range(min(p - lp, q - lm, m - lp - lm) + 1):
                            sigma = sp.zeros(m, n)
                            for k in range(lp):
                                sigma[k, k] = k + 2
                            for k in range(lm):
                                sigma[lp + k, p + k] = k + 3
                            for k in range(j):
                                sigma[lp + lm + k, lp + k] = 1
                                sigma[lp + lm + k, p + lm + k] = 1
                            gram = sigma * signature * sigma.T
                            b = sigma.T * sigma * signature
                            assert sigma.rank() == lp + lm + j
                            assert gram.rank() == lp + lm
                            assert b.rank() - (b**2).rank() == j
                            assert (b**2).rank() == (b**3).rank()
                            assert n - lp - lm - 2 * j >= 0
                            count += 1

    a, d, b, c, s, t = sp.symbols('a d b c s t', real=True)
    for h in (1, -1):
        g = sp.Matrix([[c, s], [-h * s, c]])
        signature = sp.diag(1, h)
        assert g.T * signature * g == (c**2 + h*s**2) * signature
        gram = sp.Matrix([[a, b], [b, d]])
        cross = (g.T * gram * g)[0, 1]
        assert sp.expand(cross - (b*(c**2-h*s**2)+(a-h*d)*c*s)) == 0
        if h == 1:
            tau = (d-a)/(2*b)
            equation = t**2+2*tau*t-1
            assert sp.simplify(b*(1-t**2)+(a-d)*t+b*equation) == 0
        else:
            z = -2*b/(a+d)
            equation = z*(1+t**2)-2*t
            assert sp.simplify(b*(1+t**2)+(a+d)*t+(a+d)*equation/2) == 0

    examples = [
        (sp.Matrix([[2, 1]]), sp.Matrix([[sp.sqrt(3), 0]]),
         sp.Matrix([[2, 1], [1, 2]]) / sp.sqrt(3), 1),
        (sp.Matrix([[1, 1]]), sp.Matrix([[1, 1]]), sp.eye(2), 1),
        (sp.Matrix([[2, 0, 2, 0], [0, 1, 0, 1]]),
         sp.Matrix([[1, 0, 1, 0], [0, 1, 0, 1]]),
         sp.Matrix([[sp.Rational(5,4),0,sp.Rational(3,4),0],
                    [0,1,0,0],[sp.Rational(3,4),0,sp.Rational(5,4),0],
                    [0,0,0,1]]), 2),
    ]
    for a, sigma, v, p in examples:
        signature = sp.diag(*([1]*p+[-1]*(a.cols-p)))
        assert sp.simplify(sigma*v.T-a) == sp.zeros(*a.shape)
        assert sp.simplify(v.T*signature*v-signature) == sp.zeros(a.cols)

    epsilon = sp.symbols('epsilon', positive=True)
    # j can stay constant while rank changes.
    signature = sp.diag(1, -1)
    a0, ae = sp.diag(1, 0), sp.diag(1, epsilon)
    assert a0.rank() == 1 and ae.rank() == 2
    assert a0.rank() - (a0*signature*a0.T).rank() == 0
    assert ae.rank() - (ae*signature*ae.T).rank() == 0
    # An isotropic direction in an indefinite residual changes j.
    signature = sp.diag(1, 1, -1)
    a0 = sp.Matrix([[1,0,0],[0,0,0]])
    ae = sp.Matrix([[1,0,0],[0,epsilon,epsilon]])
    assert a0.rank() - (a0*signature*a0.T).rank() == 0
    assert ae.rank() - (ae*signature*ae.T).rank() == 1
    return dict(status='passed', arithmetic='exact SymPy',
                canonical_cases=count, rotation_identities=2,
                explicit_factorizations=3, perturbation_families=2)


if __name__ == '__main__':
    result = verify()
    path = Path(__file__).resolve().parents[1] / 'results/contest/exact_identities.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
