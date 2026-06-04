from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import cholesky, eigh, qr, svd


@dataclass
class HSVDResult:
    U: np.ndarray
    Sigma: np.ndarray
    V: np.ndarray
    J: np.ndarray
    rank: int
    j: int
    pos_count: int
    neg_count: int
    residual: float
    u_unitarity: float
    v_j_unitarity: float
    reconstruction_aux: float


def signature_matrix(p: int, q: int, dtype=np.complex128) -> np.ndarray:
    return np.diag(np.concatenate([np.ones(p), -np.ones(q)])).astype(dtype)


def fro_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord="fro"))


def compact_svd(a: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vh = svd(a, full_matrices=False)
    r = int(np.sum(s > tol))
    if r == 0:
        return u[:, :0], np.zeros((0, 0), dtype=np.complex128), vh[:0, :].conj().T
    return u[:, :r], np.diag(s[:r]).astype(np.complex128), vh[:r, :].conj().T


def complete_unitary(columns: np.ndarray, m: int, tol: float = 1e-10) -> np.ndarray:
    """Complete a set of (approximately) orthonormal columns to a full m x m unitary."""
    ortho: List[np.ndarray] = []
    for i in range(columns.shape[1]):
        v = columns[:, i].astype(np.complex128).copy()
        for u in ortho:
            v = v - np.vdot(u, v) * u
        nv = np.linalg.norm(v)
        if nv > tol:
            ortho.append(v / nv)
    for i in range(m):
        if len(ortho) == m:
            break
        v = np.zeros(m, dtype=np.complex128)
        v[i] = 1.0
        for u in ortho:
            v = v - np.vdot(u, v) * u
        nv = np.linalg.norm(v)
        if nv > tol:
            ortho.append(v / nv)
    if not ortho:
        return np.eye(m, dtype=np.complex128)
    return np.column_stack(ortho)


def j_orthonormalize(
    initial_cols: np.ndarray,
    initial_signs: np.ndarray,
    J: np.ndarray,
    n: int,
    tol: float = 1e-9,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extend `initial_cols` (already J-orthonormal) to an n x n J-orthonormal basis.

    Uses indefinite (J-)Gram-Schmidt against a pool of standard and random vectors.
    Returns (V, signs) with V^* J V = diag(signs).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    cols: List[np.ndarray] = [initial_cols[:, k].copy() for k in range(initial_cols.shape[1])]
    signs: List[float] = [float(s) for s in initial_signs]

    candidate_pool: List[np.ndarray] = []
    for i in range(n):
        e = np.zeros(n, dtype=np.complex128)
        e[i] = 1.0
        candidate_pool.append(e)
    for _ in range(8 * n + 16):
        candidate_pool.append(
            (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)
        )

    idx = 0
    while len(cols) < n and idx < len(candidate_pool):
        v = candidate_pool[idx].copy()
        idx += 1
        for c, s in zip(cols, signs):
            coeff = s * (c.conj() @ J @ v)
            v = v - coeff * c
        val = float(np.real(v.conj() @ J @ v))
        nv = float(np.linalg.norm(v))
        if nv < tol or abs(val) < tol * max(1.0, nv * nv):
            continue
        sgn = 1.0 if val > 0 else -1.0
        v = v / np.sqrt(abs(val))
        cols.append(v)
        signs.append(sgn)

    V = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=np.complex128)
    return V, np.array(signs, dtype=float)


def compute_metrics(
    A: np.ndarray, U: np.ndarray, Sigma: np.ndarray, V: np.ndarray, J: np.ndarray
) -> Dict[str, float]:
    return {
        "residual": fro_error(A, U @ Sigma @ V.conj().T),
        "u_unitarity": fro_error(U.conj().T @ U, np.eye(U.shape[0], dtype=np.complex128)),
        "v_j_unitarity": fro_error(V.conj().T @ J @ V, J),
        "reconstruction_aux": fro_error(A @ J @ V @ J, U @ Sigma),
    }


def _split_indices(eigvals: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.where(eigvals > tol)[0]
    neg = np.where(eigvals < -tol)[0]
    zer = np.where(np.abs(eigvals) <= tol)[0]
    return pos, neg, zer


def hsvd_general(A: np.ndarray, J: np.ndarray, tol: float = 1e-8) -> HSVDResult:
    """Hyperbolic SVD A = U Sigma V^* with V J-unitary, U unitary.

    Algorithm (matches Algorithm 2 of the coursework):
      1. Compact SVD A = U_r Sigma_r V_r^*.
      2. Eigendecomp H := V_r^* J V_r = Q Lambda Q^*.
         Split spectrum into pos / neg / zero (tol-controlled).
      3. Non-zero part: V_+- = V_r Q_+- / sqrt(|lambda|).
         Zero part   : E = V_r Q_0, F = J V_r Sigma_r^{-2} Q_0,
            * normalize so E^* J F = I (Cholesky on the cross-Gram),
            * make F isotropic by subtracting (1/2) E (F^* J F),
            * V_v = (E+F)/sqrt(2), V_w = (E-F)/sqrt(2).
      4. Extend the J-orthonormal r + j columns to a full n x n J_eff-
         orthonormal V by indefinite Gram-Schmidt.
      5. Recover (U, Sigma) from M := A J V J_eff via QR:
            since V is J_eff-unitary, V^{-*} = J V J_eff, so
            M = A V^{-*} = U Sigma.  QR of M yields U (unitary) and Sigma
            (upper triangular with the canonical singular values on the
            diagonal up to the j-part block structure).
    """
    A = np.asarray(A, dtype=np.complex128)
    J = np.asarray(J, dtype=np.complex128)
    m, n = A.shape

    U_r, Sigma_r, V_r = compact_svd(A, tol=tol)
    r = Sigma_r.shape[0]

    if r == 0:
        U = np.eye(m, dtype=np.complex128)
        V = np.eye(n, dtype=np.complex128)
        Sigma = np.zeros((m, n), dtype=np.complex128)
        metrics = compute_metrics(A, U, Sigma, V, J)
        return HSVDResult(U, Sigma, V, J, 0, 0, 0, 0, **metrics)

    H = V_r.conj().T @ J @ V_r
    H = 0.5 * (H + H.conj().T)
    eigvals, Q = eigh(H)

    scale = max(1.0, float(np.max(np.abs(eigvals))) if eigvals.size else 1.0)
    spec_tol = max(tol, 1e-10 * scale)

    pos_idx, neg_idx, zer_idx = _split_indices(eigvals, spec_tol)
    j_count = len(zer_idx)
    pos_count = len(pos_idx)
    neg_count = len(neg_idx)

    Sigma_r_diag = np.diag(Sigma_r).astype(np.complex128)

    # --- Non-zero spectral part. ---
    nonzero_idx = np.concatenate([pos_idx, neg_idx]).astype(int)
    if nonzero_idx.size > 0:
        lam_nz = eigvals[nonzero_idx]
        Q_nz = Q[:, nonzero_idx]
        D_nz = np.sqrt(np.abs(lam_nz)).astype(np.complex128)
        V_nz = V_r @ Q_nz @ np.diag(1.0 / D_nz)
        signs_nz = np.concatenate([np.ones(pos_count), -np.ones(neg_count)])
    else:
        V_nz = np.zeros((n, 0), dtype=np.complex128)
        signs_nz = np.zeros(0)

    # --- Zero spectral part: build (V_v, V_w) pairs. ---
    if j_count > 0:
        Q_zer = Q[:, zer_idx]
        E = V_r @ Q_zer                                       # n x j
        Sigma_inv2 = np.diag(1.0 / (Sigma_r_diag ** 2))
        F = J @ V_r @ Sigma_inv2 @ Q_zer                      # n x j

        cross = E.conj().T @ J @ F                            # j x j HPD
        cross = 0.5 * (cross + cross.conj().T)
        L = cholesky(cross, lower=True)
        L_inv = np.linalg.inv(L)
        E2 = E @ L_inv.conj().T                               # E2^* J F2 = I
        F2 = F @ L_inv

        K = F2.conj().T @ J @ F2                              # j x j Hermitian
        K = 0.5 * (K + K.conj().T)
        F_iso = F2 - 0.5 * E2 @ K                             # F_iso^* J F_iso ≈ 0

        V_v = (E2 + F_iso) / np.sqrt(2.0)
        V_w = (E2 - F_iso) / np.sqrt(2.0)
        signs_v = np.ones(j_count)
        signs_w = -np.ones(j_count)
    else:
        V_v = np.zeros((n, 0), dtype=np.complex128)
        V_w = np.zeros((n, 0), dtype=np.complex128)
        signs_v = np.zeros(0)
        signs_w = np.zeros(0)

    # --- Assemble J-orthonormal initial block.  Ordering: V_+ , V_v , V_- , V_w ---
    V_pos = V_nz[:, :pos_count]
    V_neg = V_nz[:, pos_count:]
    V_init = np.concatenate([V_pos, V_v, V_neg, V_w], axis=1)
    signs_init = np.concatenate([np.ones(pos_count), signs_v, -np.ones(neg_count), signs_w])

    # --- Extend to full n x n via indefinite Gram-Schmidt. ---
    rng = np.random.default_rng(12345)
    V, signs_full = j_orthonormalize(V_init, signs_init, J, n, tol=tol, rng=rng)
    if V.shape[1] < n:
        # Last-ditch: pad with orthonormal complement vectors w.r.t. existing cols,
        # accepting that V may fail to be exactly J-unitary.
        Q_pad, _ = qr(V, mode="complete")
        extra_needed = n - V.shape[1]
        extras = Q_pad[:, V.shape[1]:V.shape[1] + extra_needed]
        V = np.concatenate([V, extras], axis=1)
        signs_full = np.concatenate([signs_full, np.ones(extra_needed)])

    # Permute V columns so that V^* J V == J exactly (positive signs first,
    # negative signs last, matching J = diag(I_p, -I_q)).
    target_signs = np.real(np.diag(J))
    perm = np.argsort(-signs_full, kind="stable")             # +1 entries first
    V = V[:, perm]
    signs_full = signs_full[perm]
    # If the permuted inertia matches J exactly we are done; otherwise we
    # report J_eff = diag(signs_full) as the effective signature.
    if signs_full.shape[0] == target_signs.shape[0] and np.allclose(signs_full, target_signs):
        J_eff = J.copy()
    else:
        J_eff = np.diag(signs_full).astype(np.complex128)

    # --- Recover U and Sigma via QR of M = A J V J_eff = U Sigma. ---
    M = A @ J @ V @ J_eff
    if m >= n:
        Q_u, R_u = qr(M, mode="economic")
        Sigma = np.zeros((m, n), dtype=np.complex128)
        Sigma[:n, :n] = R_u
        U = complete_unitary(Q_u, m, tol=tol)
    else:
        Q_u, R_u = qr(M, mode="full")
        U = Q_u
        Sigma = R_u

    metrics = compute_metrics(A, U, Sigma, V, J_eff)
    return HSVDResult(
        U=U,
        Sigma=Sigma,
        V=V,
        J=J_eff,
        rank=r,
        j=j_count,
        pos_count=pos_count,
        neg_count=neg_count,
        **metrics,
    )


def make_random_problem(
    m: int, n: int, p: int, complex_case: bool = True, seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if complex_case:
        A = rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))
    else:
        A = rng.standard_normal((m, n))
    q = n - p
    J = signature_matrix(p, q)
    return A.astype(np.complex128), J


def make_rank_deficient_problem(
    m: int, n: int, p: int, rank: int, seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((m, rank)) + 1j * rng.standard_normal((m, rank))
    R = rng.standard_normal((n, rank)) + 1j * rng.standard_normal((n, rank))
    A = L @ R.conj().T
    J = signature_matrix(p, n - p)
    return A.astype(np.complex128), J


def make_degenerate_problem(
    m: int, n: int, p: int, j_target: int, seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Build A whose H = V_r^* J V_r has approximately j_target zero eigenvalues.

    We construct V_r explicitly so that its J-Gram has the prescribed inertia.
    Pick (n-p) = q.  Choose a J-orthonormal frame [v_+ (r-j_target-?) , v_- (?), v_iso (j_target)]
    where v_iso are isotropic vectors with v^* J v = 0.  Then V_r := those columns.
    """
    rng = np.random.default_rng(seed)
    q = n - p
    J = signature_matrix(p, q)

    r = min(m, n)
    j_target = min(j_target, min(p, q, r // 2))
    n_pos = (r - 2 * j_target) // 2 + max(0, (r - 2 * j_target) % 2)
    n_neg = r - 2 * j_target - n_pos
    n_pos = max(n_pos, 0)
    n_neg = max(n_neg, 0)

    # Positive-signature columns: standard basis vectors in the +1 part of J.
    cols: List[np.ndarray] = []
    pos_avail = list(range(p))
    neg_avail = list(range(p, n))
    rng.shuffle(pos_avail)
    rng.shuffle(neg_avail)

    for _ in range(n_pos):
        if not pos_avail:
            break
        i = pos_avail.pop()
        e = np.zeros(n, dtype=np.complex128)
        e[i] = 1.0
        cols.append(e)
    for _ in range(n_neg):
        if not neg_avail:
            break
        i = neg_avail.pop()
        e = np.zeros(n, dtype=np.complex128)
        e[i] = 1.0
        cols.append(e)
    # Isotropic vectors: (e_p + e_q)/sqrt(2) for one + and one - index.
    for _ in range(j_target):
        if not pos_avail or not neg_avail:
            break
        ip = pos_avail.pop()
        iq = neg_avail.pop()
        v = np.zeros(n, dtype=np.complex128)
        v[ip] = 1.0 / np.sqrt(2.0)
        v[iq] = 1.0 / np.sqrt(2.0)
        cols.append(v)
        # Add a second isotropic vector in the same hyperbolic plane to keep V_r columns linearly independent.
        v2 = np.zeros(n, dtype=np.complex128)
        v2[ip] = 1.0 / np.sqrt(2.0)
        v2[iq] = -1.0 / np.sqrt(2.0)
        cols.append(v2)

    while len(cols) < r and (pos_avail or neg_avail):
        if pos_avail:
            i = pos_avail.pop()
        else:
            i = neg_avail.pop()
        e = np.zeros(n, dtype=np.complex128)
        e[i] = 1.0
        cols.append(e)

    V_r = np.column_stack(cols[:r])

    # Random orthogonal mixing on the columns (preserves rank, but breaks J-frame
    # exact zeros only mildly).  Multiply A from left by random unitary too.
    Sigma_r = np.diag(np.sort(rng.uniform(1.0, 3.0, size=r))[::-1]).astype(np.complex128)
    U_full = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    U_full, _ = qr(U_full)
    U_r = U_full[:, :r]

    A = U_r @ Sigma_r @ V_r.conj().T
    return A.astype(np.complex128), J
