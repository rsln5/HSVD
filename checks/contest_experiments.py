"""Reproduce the tables in output/pdf/nirs_computer_science.tex, from the project root.

Run: python3 -m checks.contest_experiments
NumPy, threadpoolctl and mpmath are required in addition to the local package.
The Jacobi baseline is our Python prototype, not the authors' AccJac code.
It is compared only on numerically full-column-rank inputs; the general
algorithm is also run on all inputs outside that restricted domain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter

THREAD_VARIABLES = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
# Set before importing NumPy in the CLI, without changing the caller's
# environment when tests or another script import the Jacobi baseline.
if __name__ == "__main__":
    for _variable in THREAD_VARIABLES:
        os.environ[_variable] = "1"

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from examples.cases import generated_case, standard_cases
from hsvd import NumericalBreakdownError, hsvd
from hsvd._numeric import component_scale, divide_by_real, euclidean_gram

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "contest"
PROTOCOL = "contest-v5"
SOURCE_PATHS = (
    "hsvd/core.py", "hsvd/_numeric.py", "hsvd/_types.py", "hsvd/__init__.py",
    "examples/cases.py", "checks/contest_experiments.py",
    "checks/render_contest_tables.py", "pyproject.toml",
)


def jacobi_applicability(a):
    """Record the baseline's domain without substituting it for general HSVD.

    This numerical rank test is a reproducible eligibility criterion, not a
    proof of the exact rank of an arbitrary floating-point input.
    """
    a = np.asarray(a)
    if a.ndim != 2 or min(a.shape) == 0:
        raise ValueError("Expected a nonempty matrix")
    m, n = a.shape
    if m < n:
        return dict(applicable=False, reason="Fewer rows than columns", columns=n)
    scale = component_scale(a)
    singulars = np.linalg.svd(divide_by_real(a, scale), compute_uv=False) if scale else np.zeros(n)
    threshold = float(100 * max(m, n) * np.finfo(float).eps * singulars[0])
    rank = int(np.count_nonzero(singulars > threshold))
    return dict(applicable=rank == n, numerical_rank=rank, columns=n,
                rank_threshold_scaled=threshold,
                reason=None if rank == n else "Numerical full column rank required")


def factor_metrics(a, p, u, sigma, v):
    """Evaluate the returned factors on a common scale, including Sigma.

    Direct Frobenius norms of a can overflow or underflow although every
    entry is finite. Scaling both numerator and denominator preserves the
    relative residual while avoiding this loss of the validation check.
    """
    m, n = a.shape
    signs = np.r_[np.ones(p), -np.ones(n-p)]
    scale = component_scale(a)
    a0 = divide_by_real(a, scale) if scale else a
    sigma0 = sigma / scale if scale else sigma
    norm = np.linalg.norm(a0, "fro")
    us = u @ sigma0
    us_norm = np.linalg.norm(us, "fro")
    return dict(
        reconstruction_error=float(np.linalg.norm(a0-us @ v.conj().T, "fro") / (norm or 1)),
        unitarity_error=float(np.linalg.norm(euclidean_gram(u)-np.eye(m), "fro") / np.sqrt(m)),
        j_unitarity_error=float(np.linalg.norm(v.conj().T @ (signs[:, None]*v)-np.diag(signs), "fro") / np.sqrt(n)),
        transform_error=float(np.linalg.norm(((a0*signs) @ v)*signs-us, "fro") / (us_norm or 1)),
        norm_v=float(np.linalg.norm(v, 2)),
    )


def jacobi_hsvd(a, p, *, tol=1e-13, max_sweeps=100, check_tol=1e-9):
    """Full-column-rank real/complex baseline with full U, Sigma, V.

    Right phases make x* y positive real before each plane rotation.
    W accumulates A W; the returned right factor is J W J.
    """
    a = np.asarray(a)
    if a.ndim != 2 or min(a.shape) == 0 or a.dtype.kind not in "biufc":
        raise ValueError("Expected a nonempty numeric matrix")
    m, n = a.shape
    if (m < n or isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral)
            or not 0 <= p <= n or isinstance(max_sweeps, (bool, np.bool_))
            or not isinstance(max_sweeps, Integral) or max_sweeps < 1):
        raise ValueError("Invalid dimensions or iteration parameters")
    tolerances = {}
    for name, value in (("tol", tol), ("check_tol", check_tol)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not 0 < value < 1):
            raise ValueError(f"{name} must be finite and in (0, 1)")
        tolerances[name] = float(value)
        if not 0 < tolerances[name] < 1:
            raise ValueError(f"{name} must remain in (0, 1) in double precision")
    dtype = np.complex128 if np.iscomplexobj(a) else np.float64
    with np.errstate(over="ignore", invalid="ignore"):
        a = np.array(a, dtype=dtype, copy=True)
    if not np.isfinite(a).all():
        raise ValueError("A must be finite and representable in double precision")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _jacobi_compute(a, int(p), tolerances["tol"], int(max_sweeps),
                                   tolerances["check_tol"])
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise NumericalBreakdownError(
            f"Jacobi arithmetic failed: {exc}", stage="arithmetic") from exc


def _jacobi_compute(a, p, tol, max_sweeps, check_tol):
    m, n = a.shape
    scale = component_scale(a)
    if scale == 0:
        raise ValueError("Full column rank required")
    work = divide_by_real(a, scale)
    if not jacobi_applicability(a)["applicable"]:
        raise ValueError("Numerical full column rank required")
    signs = np.r_[np.ones(p), -np.ones(n-p)]
    w = np.eye(n, dtype=a.dtype)
    rotations = 0
    for sweep in range(1, max_sweeps+1):
        changed = 0
        for i in range(n):
            for k in range(i+1, n):
                aa = float(np.vdot(work[:,i],work[:,i]).real)
                dd = float(np.vdot(work[:,k],work[:,k]).real)
                beta = np.vdot(work[:,i], work[:,k])
                b = abs(beta)
                if b <= tol*np.sqrt(aa)*np.sqrt(dd):
                    continue
                phase = np.conj(beta)/b
                work[:,k] *= phase
                w[:,k] *= phase
                h = signs[i]*signs[k]
                if h > 0:
                    tau = (dd-aa)/(2*b)
                    t = np.copysign(1., tau)/(abs(tau)+np.hypot(1.,tau))
                    c = 1/np.hypot(1.,t)
                else:
                    z = -2*b/(aa+dd)
                    if not abs(z) < 1:
                        raise NumericalBreakdownError("Hyperbolic pivot unresolved", stage="pivot")
                    t = z/(1+np.sqrt((1-z)*(1+z)))
                    c = 1/np.sqrt((1-t)*(1+t))
                s = c*t
                for matrix in (work,w):
                    x,y = matrix[:,i].copy(),matrix[:,k].copy()
                    matrix[:,i] = c*x-h*s*y
                    matrix[:,k] = s*x+c*y
                changed += 1
        rotations += changed
        # Test the actual end-of-sweep columns. Requiring a further sweep
        # with no rotations would reject an already converged final sweep.
        norms = np.linalg.norm(work, axis=0)
        normalized = work / norms
        correlations = euclidean_gram(normalized)
        np.fill_diagonal(correlations, 0)
        if np.max(np.abs(correlations)) <= tol:
            break
    else:
        raise NumericalBreakdownError("Jacobi did not converge within max_sweeps", stage="iteration")
    norms = np.linalg.norm(work,axis=0)
    order = np.r_[np.argsort(-norms[:p]),p+np.argsort(-norms[p:])]
    work,w,norms = work[:,order],w[:,order],norms[order]
    thin = work/norms
    if m > n:
        _,_,vh = np.linalg.svd(thin.conj().T,full_matrices=True)
        u = np.column_stack((thin,vh.conj().T[:,n:]))
    else:
        u = thin
    sigma = np.zeros((m,n))
    sigma[np.arange(n),np.arange(n)] = norms*scale
    v = signs[:,None]*w*signs
    if not all(np.isfinite(x).all() for x in (u, sigma, v)):
        raise NumericalBreakdownError("A Jacobi factor is nonfinite", stage="arithmetic")
    metrics = factor_metrics(a,p,u,sigma,v)
    if (not all(np.isfinite(value) for value in metrics.values())
            or max(metrics[k] for k in metrics if k != "norm_v") > check_tol):
        raise NumericalBreakdownError(
            "Jacobi factor verification failed", stage="verification", diagnostics=metrics)
    return u,sigma,v,metrics,dict(sweeps=sweep,rotations=rotations)


def run_method(a,p,method,**kwargs):
    """Run one full-factor call; retain failures and its computation time.

    The eligibility test and common reporting metrics are outside the timer.
    Each method's own rank decisions, completion and verification are inside.
    """
    if method not in ("general", "jacobi"):
        raise ValueError(f"Unknown method: {method}")
    if method == "jacobi":
        applicability = jacobi_applicability(a)
        if not applicability["applicable"]:
            return dict(status="not_applicable", stage="domain",
                        reason=applicability["reason"], applicability=applicability,
                        elapsed_seconds=None)
    start = perf_counter()
    try:
        if method == "general":
            result = hsvd(a,p,**kwargs)
            elapsed = perf_counter() - start
            st=result.structure
            values=np.r_[result.Sigma[np.arange(st.l_plus),np.arange(st.l_plus)],
                         result.Sigma[st.l_plus+np.arange(st.l_minus),p+np.arange(st.l_minus)]]
            metrics = asdict(result.diagnostics)
            metrics.update(factor_metrics(a,p,result.U,result.Sigma,result.V))
            if not all(np.isfinite(value) for value in metrics.values()):
                raise NumericalBreakdownError("Nonfinite common diagnostics", stage="verification",
                                              diagnostics=metrics)
            return dict(status="ok",structure=asdict(st),metrics=metrics,
                        hyperbolic_values=values.tolist(), elapsed_seconds=elapsed)
        _,sigma,_,metrics,iterations = jacobi_hsvd(a,p,**kwargs)
        elapsed = perf_counter() - start
        return dict(status="ok",metrics=metrics,hyperbolic_values=np.diag(sigma).tolist(),
                    elapsed_seconds=elapsed, applicability=applicability, **iterations)
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
        result = dict(status="failed",reason=str(exc),stage=getattr(exc,"stage",None),
                      elapsed_seconds=perf_counter()-start)
        diagnostic = getattr(exc,"diagnostics",None)
        if diagnostic is not None:
            metrics = diagnostic if isinstance(diagnostic, dict) else asdict(diagnostic)
            result["metrics"] = {k: value if np.isfinite(value) else None
                                 for k, value in metrics.items()}
            nonfinite = [k for k, value in metrics.items() if not np.isfinite(value)]
            if nonfinite:
                result["nonfinite_metrics"] = nonfinite
        return result


def _expected_structure(case):
    return dict(rank=case.l_plus+case.l_minus+case.j,
                l_plus=case.l_plus, l_minus=case.l_minus, j=case.j,
                p0=case.p-case.l_plus-case.j,
                q0=case.A.shape[1]-case.p-case.l_minus-case.j)


def _validate_known_case(case, result):
    """Check structure and values against the independent canonical generator."""
    if result["status"] != "ok":
        raise AssertionError((case.name, result))
    expected = _expected_structure(case)
    if "structure" in result and result["structure"] != expected:
        raise AssertionError((case.name, expected, result["structure"]))
    values = np.r_[sorted(case.positive_values, reverse=True),
                   sorted(case.negative_values, reverse=True)]
    np.testing.assert_allclose(result["hyperbolic_values"], values,
                               atol=2e-10, rtol=2e-10)
    return expected


def _check_reference(case, result):
    """Preserve a wrong successful answer as a reference failure in the log."""
    if result["status"] == "ok":
        try:
            _validate_known_case(case, result)
        except AssertionError as exc:
            result.update(status="failed", stage="reference", reason=str(exc))
    return result


def timed_case(case, method, repetitions):
    """Warm up once, then retain every measured attempt, including failures."""
    if repetitions < 1:
        raise ValueError("At least one timing repetition is required")
    result = _check_reference(case, run_method(case.A, case.p, method))
    result["warmup_seconds"] = result["elapsed_seconds"]
    result["time_seconds"] = []
    result["attempts"] = []
    if result["status"] != "ok":
        result["median_ms"] = None
        return result
    for repetition in range(repetitions):
        measured = _check_reference(case, run_method(case.A, case.p, method))
        attempt = dict(repetition=repetition, status=measured["status"],
                       elapsed_seconds=measured["elapsed_seconds"])
        if measured["status"] != "ok":
            attempt.update(reason=measured.get("reason"), stage=measured.get("stage"))
            result.update(status="failed", reason=measured.get("reason"),
                          stage=measured.get("stage"), failed_repetition=repetition)
        else:
            result["time_seconds"].append(measured["elapsed_seconds"])
            for key, value in measured["metrics"].items():
                result["metrics"][key] = max(result["metrics"][key], value)
        result["attempts"].append(attempt)
    result["median_ms"] = (float(np.median(result["time_seconds"])*1000)
                           if result["status"] == "ok" else None)
    return result


def high_precision_reference(a, p, *, digits=80):
    """Independent HSVD values from the actual binary input, at high precision.

    Only the full-column-rank conditioning examples use this oracle. Their
    nonzero inertia is (p,n-p), while the remaining m-n eigenvalues are zero.
    Matrix products and the Hermitian eigenproblem are evaluated by mpmath,
    without first forming a rounded NumPy Gram matrix.
    """
    import mpmath as mp

    a = np.asarray(a)
    if not jacobi_applicability(a)["applicable"]:
        raise ValueError("The conditioning reference requires numerical full column rank")
    m, n = a.shape
    if not 0 <= p <= n:
        raise ValueError("Invalid signature")
    with mp.workdps(digits):
        aa = mp.matrix([[mp.mpc(mp.mpf(float(z.real)), mp.mpf(float(z.imag)))
                         if np.iscomplexobj(a) else mp.mpf(float(z)) for z in row] for row in a])
        j = mp.diag([1 if k < p else -1 for k in range(n)])
        gram = aa * j * aa.H
        eig = mp.eighe if np.iscomplexobj(a) else mp.eigsy
        eigenvalues = list(eig(gram, eigvals_only=True))
        zero_tol = 100 * m * mp.eps * max(abs(x) for x in eigenvalues)
        positive = sorted((x for x in eigenvalues if x > zero_tol), reverse=True)
        negative = sorted((-x for x in eigenvalues if x < -zero_tol), reverse=True)
        if len(positive) != p or len(negative) != n-p:
            raise ArithmeticError("High-precision reference did not resolve the expected inertia")
        values = [mp.sqrt(x) for x in positive + negative]
        return dict(method=f"mpmath.{eig.__name__}(A J A*)", mpmath_version=mp.__version__,
                    decimal_digits=digits, source="actual stored binary64 input entries",
                    hyperbolic_values=[mp.nstr(x, digits) for x in values],
                    positive_count=len(positive), negative_count=len(negative))


def reference_relative_error(computed, reference):
    """Compare binary64 results with the decimal strings of the MP oracle."""
    import mpmath as mp

    if len(computed) != len(reference["hyperbolic_values"]):
        raise ValueError("Incorrect number of hyperbolic values")
    with mp.workdps(reference["decimal_digits"]):
        errors = [abs(mp.mpf(float(x)) - mp.mpf(y)) / mp.mpf(y)
                  for x, y in zip(computed, reference["hyperbolic_values"])]
        return float(max(errors, default=mp.mpf(0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5,
                        help="Number of timed full-factor calls after one warm-up (default: 5)")
    parser.add_argument("--output", type=Path, default=OUT,
                        help="Protocol directory (use a temporary directory for trial runs)")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    source_hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    data = dict(protocol=PROTOCOL,created_utc=datetime.now(timezone.utc).isoformat(),
                environment=dict(python=sys.version,numpy=np.__version__,
                platform=platform.platform(),machine=platform.machine()),
                settings=dict(rank_rtol="100*max(m,n)*eps",spectral_rtol="100*max(m,n)*eps",
                check_tol=1e-9,jacobi_tol=1e-13,max_sweeps=100,
                timing_repetitions=args.repetitions,blas_threads_requested=1,
                jacobi_implementation="in-project Python prototype; not AccJac or optimized vendor code",
                jacobi_domain="m >= n and numerical rank n under the recorded SVD rank threshold",
                comparison_signatures="p in {0,n/2,n}; real and complex; three seeds",
                timing_scope="full-factor call including internal validation and diagnostics; excludes eligibility, common reporting metrics, reference checks and serialization",
                timing_aggregation="median of per-input median times; eighteen inputs per size",
                timing_method_order="alternate general-first and jacobi-first between inputs",
                timing_failure_policy="retain every measured attempt; no median reported if any attempt fails",
                generator_rotation_size=0.4,comparison_seeds=[17,31,2026],
                conditioning_reference="80-digit mpmath eigendecomposition of A J A* formed from actual binary64 entries; generator errors reported separately"),
                structural=[],comparison=[],conditioning=[],thresholds=[])
    inputs = {}
    with threadpool_limits(limits=1):
        data["environment"]["threadpools"] = threadpool_info()
        data["environment"]["thread_environment"] = {k:os.environ.get(k) for k in THREAD_VARIABLES}
        data["environment"]["threadpool_note"] = (
            "threadpoolctl reports the pools below; requested thread limits are recorded separately"
            if data["environment"]["threadpools"] else
            "threadpoolctl did not identify a BLAS pool; the effective thread count is unverified")
        for idx,case in enumerate(standard_cases()):
            inputs[f"structural_{idx}"] = case.A
            result = _check_reference(case, run_method(case.A,case.p,"general"))
            expected = _expected_structure(case)
            baseline = _check_reference(case, run_method(case.A,case.p,"jacobi"))
            data["structural"].append(dict(name=case.name,input_key=f"structural_{idx}",shape=list(case.A.shape),
                group="generated" if case.name.startswith("generated_") else "analytic",
                p=case.p,complex_data=bool(np.iscomplexobj(case.A)),expected=expected,
                jacobi=baseline,**result))
        for m,n in [(8,8),(16,8),(32,16),(64,32)]:
            for p,complex_data,seed in ((p,c,s) for p in (0,n//2,n)
                                        for c in (False,True) for s in (17,31,2026)):
                case=generated_case(m,p,n-p,p,n-p,0,seed=seed,complex_data=complex_data)
                key=f"comparison_{m}_{n}_{p}_{int(complex_data)}_{seed}"
                inputs[key]=case.A
                order=("general","jacobi") if len(data["comparison"]) % 2 == 0 else ("jacobi","general")
                row=dict(shape=[m,n],p=p,complex_data=complex_data,seed=seed,input_key=key,
                         expected=_expected_structure(case),method_order=list(order),
                         applicability=jacobi_applicability(case.A))
                for method in order:
                    row[method]=timed_case(case,method,args.repetitions)
                data["comparison"].append(row)
        for ratio in [1,1e2,1e4,1e6,1e8]:
            values=np.geomspace(1,1/ratio,4)
            case=generated_case(16,4,4,4,4,0,seed=2026,positive=values,negative=values)
            key=f"conditioning_{ratio:g}"
            inputs[key]=case.A
            row=dict(sigma_ratio=ratio,input_key=key,p=4,shape=list(case.A.shape),
                     complex_data=bool(np.iscomplexobj(case.A)),
                     applicability=jacobi_applicability(case.A),
                     kappa_A=float(np.linalg.cond(case.A)))
            row["reference"] = high_precision_reference(case.A, 4)
            expected=np.r_[values,values]
            row["generator_values"] = expected.tolist()
            row["generator_vs_stored_input_relative_error"] = reference_relative_error(expected, row["reference"])
            for method in ["general","jacobi"]:
                row[method]=run_method(case.A,4,method)
                if row[method]["status"] == "ok":
                    computed=np.array(row[method]["hyperbolic_values"])
                    if computed.shape != expected.shape:
                        row[method].update(status="failed", stage="reference",
                                           reason="Number of hyperbolic values differs from the generator")
                    else:
                        row[method]["max_relative_value_error_generator"]=float(np.max(np.abs(computed-expected)/expected))
                        row[method]["max_relative_value_error_stored_input"] = reference_relative_error(computed, row["reference"])
            data["conditioning"].append(row)
        for delta in [0.,1e-4,1e-8,1e-12,1e-15]:
            a=np.array([[1.,1.+delta]])
            key=f"threshold_{delta:g}"
            inputs[key]=a
            for tau in [1e-10,1e-14,1e-18]:
                data["thresholds"].append(dict(delta=delta,input_key=key,p=1,shape=list(a.shape),
                    effective_delta=float(a[0,1]-1),
                    exact_j_for_stored_input=int(a[0,1] == 1),
                    spectral_rtol=tau,A=a.tolist(),
                    jacobi=run_method(a,1,"jacobi"),
                    **run_method(a,1,"general",spectral_rtol=tau)))
    if source_hashes != {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}:
        raise RuntimeError("Sources changed during the experiment; no protocol was saved")
    data["sha256"] = source_hashes
    np.savez_compressed(out/"inputs.npz",**inputs)
    data["inputs_sha256"] = hashlib.sha256((out/"inputs.npz").read_bytes()).hexdigest()
    (out/"results.json").write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(structural=len(data["structural"]),comparison=len(data["comparison"]),
          conditioning=data["conditioning"],thresholds=data["thresholds"]),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
