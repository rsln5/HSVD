"""Constructive full HSVD, including isotropic Jordan blocks.

The stages follow output/pdf/nirs_computer_science.tex. All ranks and inertias
reported by this floating-point implementation are numerical estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike

from ._numeric import (
    NumericArray,
    RealArray,
    component_scale,
    divide_by_real,
    euclidean_gram,
    frobenius_norm,
    hermitian_part,
    signature_signs,
)
from ._types import (
    HSVDDiagnostics,
    HSVDResult,
    HSVDStructure,
    NumericalBreakdownError,
)

# Conservative rounding margin for rank/inertia decisions, not an error bound.
DEFAULT_ROUNDING_MARGIN = 100.0
DEFAULT_CHECK_TOLERANCE = 1e-9


@dataclass(frozen=True)
class _TruncatedSVD:
    """Retained SVD bases, full left completion, and truncation diagnostics."""

    left_basis: NumericArray
    left_complement: NumericArray
    right_basis: NumericArray
    singular_values: RealArray
    rank_threshold: float
    truncation_error: float

    @property
    def rank(self) -> int:
        return self.right_basis.shape[1]

    @property
    def retained_values(self) -> RealArray:
        return self.singular_values[:self.rank]


@dataclass(frozen=True)
class _SpectralBlocks:
    """Eigenvectors split by sign; positive/negative values are square roots."""

    eigenvalues: RealArray
    positive_vectors: NumericArray
    negative_vectors: NumericArray
    zero_vectors: NumericArray
    positive_values: RealArray
    negative_values: RealArray
    threshold: float
    structure: HSVDStructure


@dataclass(frozen=True)
class _RightBasisBlocks:
    """Nonzero right vectors X+, X- and normalized neutral chains E, G."""

    positive: NumericArray
    negative: NumericArray
    neutral: NumericArray
    dual: NumericArray


def hsvd(
    A: ArrayLike,
    p: int,
    *,
    rank_rtol: float | None = None,
    spectral_rtol: float | None = None,
    check_tol: float = DEFAULT_CHECK_TOLERANCE,
) -> HSVDResult:
    """Compute full HSVD A ~= U Sigma V* for J = diag(I_p, -I_(n-p)).

    Parameters
    ----------
    A : array_like, shape (m, n)
        Finite nonempty real/complex matrix. Computation uses double precision;
        the input is not modified.
    p : int
        Number of leading +1 entries of J (0 <= p <= n).
    rank_rtol, spectral_rtol : float in [0, 1), optional
        Relative cutoffs. Defaults: 100 * max(m, n) * machine epsilon.
        On the scaled input, retain s_i > rank_rtol*s_max; classify eigenvalues
        of C as zero when |lambda| <= spectral_rtol*s_max**2.
        Set a smaller cutoff only when justified by the data/error model.
    check_tol : float in (0, 1)
        Upper bound for all four final factor residuals. These checks cannot be
        disabled. They validate this output, not the exact rank or forward
        accuracy of individual singular vectors.

    Returns
    -------
    HSVDResult
        Full factors U:(m,m), Sigma:(m,n), V:(n,n), structure and diagnostics.

    Raises
    ------
    ValueError, TypeError
        Invalid data or parameters.
    NumericalBreakdownError
        Inconsistent numerical structure, unresolved basis, overflow, or
        failure of the final factor identities. For final residual failures,
        the exception's diagnostics attribute contains the measured errors.
    """
    matrix, p = _validate_input(A, p)
    default_tolerance = _default_relative_tolerance(matrix.shape)
    rank_tolerance = _validate_tolerance(rank_rtol, default_tolerance, "rank_rtol")
    spectral_tolerance = _validate_tolerance(
        spectral_rtol, default_tolerance, "spectral_rtol"
    )
    check_tolerance = _validate_tolerance(
        check_tol, DEFAULT_CHECK_TOLERANCE, "check_tol", allow_zero=False
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _compute_factorization(
                matrix, p, rank_tolerance, spectral_tolerance,
                default_tolerance, check_tolerance,
            )
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise NumericalBreakdownError(
            f"Floating-point factorization failed: {exc}", stage="arithmetic"
        ) from exc


def _default_relative_tolerance(shape: tuple[int, int]) -> float:
    """Scale the rounding margin by matrix dimension and double precision."""
    return float(DEFAULT_ROUNDING_MARGIN * max(shape) * np.finfo(np.float64).eps)


def _validate_input(matrix: ArrayLike, p: int) -> tuple[NumericArray, int]:
    """Validate dimensions/signature and copy data into double precision."""
    try:
        array = np.asarray(matrix)
    except ValueError as exc:
        raise ValueError("A must be a rectangular two-dimensional matrix") from exc
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError("A must be a nonempty two-dimensional matrix")
    if array.dtype.kind not in "biufc":
        raise TypeError("A must contain real or complex numbers")
    if isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral):
        raise ValueError("p must be an integer, not a boolean")
    if not 0 <= p <= array.shape[1]:
        raise ValueError("p must satisfy 0 <= p <= A.shape[1]")
    dtype = np.complex128 if np.iscomplexobj(array) else np.float64
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(array, dtype=dtype, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("A must be finite and representable in double precision")
    return array, int(p)


def _validate_tolerance(
    value: float | None, default: float, name: str, *, allow_zero: bool = True
) -> float:
    """Resolve an optional tolerance and enforce its finite admissible range."""
    value = default if value is None else value
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not 0 <= value < 1
        or (not allow_zero and value == 0)
    ):
        interval = "[0, 1)" if allow_zero else "(0, 1)"
        raise ValueError(f"{name} must be a finite real number in {interval}")
    resolved = float(value)
    if resolved >= 1:
        raise ValueError(f"{name} must remain less than 1 in double precision")
    if not allow_zero and resolved == 0:
        raise ValueError(f"{name} must remain positive in double precision")
    return resolved


def _compute_factorization(
    matrix: NumericArray,
    p: int,
    rank_tolerance: float,
    spectral_tolerance: float,
    basis_tolerance: float,
    check_tolerance: float,
) -> HSVDResult:
    """Run the construction on validated data and verify the returned factors."""
    scale = component_scale(matrix)
    if scale == 0:
        return _zero_factorization(matrix, p)
    scaled_matrix = divide_by_real(matrix, scale)
    signs = signature_signs(matrix.shape[1], p)
    svd = _truncated_svd(scaled_matrix, rank_tolerance)
    spectrum = _decompose_compressed_form(svd, signs, p, spectral_tolerance)
    right_blocks = _build_right_blocks(svd, spectrum, signs)
    complement = _complete_right_basis(
        right_blocks, signs, spectrum.structure, basis_tolerance, check_tolerance
    )
    left_factor, canonical, right_factor = _assemble_factors(
        svd, spectrum, right_blocks, complement, scale, p
    )
    factors = (left_factor, canonical, right_factor)
    if not all(np.isfinite(factor).all() for factor in factors):
        raise NumericalBreakdownError(
            "A factor contains nonfinite entries", stage="arithmetic"
        )
    diagnostics = _calculate_diagnostics(
        scaled_matrix, scale, left_factor, canonical, right_factor,
        signs, svd.truncation_error,
    )
    _check_diagnostics(diagnostics, check_tolerance)
    return HSVDResult(
        U=left_factor,
        Sigma=canonical,
        V=right_factor,
        p=p,
        structure=spectrum.structure,
        diagnostics=diagnostics,
        input_scale=scale,
        singular_values_scaled=svd.singular_values,
        eigenvalues_scaled=spectrum.eigenvalues,
        rank_threshold_scaled=svd.rank_threshold,
        spectral_threshold_scaled=spectrum.threshold,
    )


def _zero_factorization(matrix: NumericArray, p: int) -> HSVDResult:
    """Construct exact factors for zero input without spectral computations."""
    rows, columns = matrix.shape
    return HSVDResult(
        U=np.eye(rows, dtype=matrix.dtype),
        Sigma=np.zeros((rows, columns)),
        V=np.eye(columns, dtype=matrix.dtype),
        p=p,
        structure=HSVDStructure(rank=0, l_plus=0, l_minus=0, j=0, p0=p, q0=columns - p),
        diagnostics=HSVDDiagnostics(
            reconstruction_error=0.0,
            unitarity_error=0.0,
            j_unitarity_error=0.0,
            transform_error=0.0,
            norm_v=1.0,
            rank_truncation_error=0.0,
        ),
        input_scale=0.0,
        singular_values_scaled=np.zeros(min(rows, columns)),
        eigenvalues_scaled=np.empty(0),
        rank_threshold_scaled=0.0,
        spectral_threshold_scaled=0.0,
    )


def _truncated_svd(matrix: NumericArray, rank_tolerance: float) -> _TruncatedSVD:
    """Select a numerical rank and measure the relative discarded SVD tail."""
    rows, columns = matrix.shape
    # Reduced SVD of a wide matrix already supplies the full left basis;
    # constructing its unused full right basis would waste time and memory.
    left, singular_values, right_adjoint = np.linalg.svd(
        matrix, full_matrices=(rows >= columns)
    )
    rank_threshold = rank_tolerance * singular_values[0]
    rank = int(np.count_nonzero(singular_values > rank_threshold))
    if rank == 0:
        # Nonzero scaled input and rank_tolerance < 1 must retain s_max.
        raise NumericalBreakdownError(
            "No nonzero singular value was resolved", stage="rank"
        )
    truncation_error = (
        frobenius_norm(singular_values[rank:]) / frobenius_norm(singular_values)
    )
    return _TruncatedSVD(
        left_basis=left[:, :rank],
        left_complement=left[:, rank:],
        right_basis=right_adjoint.conj().T[:, :rank],
        singular_values=singular_values,
        rank_threshold=float(rank_threshold),
        truncation_error=truncation_error,
    )


def _decompose_compressed_form(
    svd: _TruncatedSVD, signs: RealArray, p: int, spectral_tolerance: float
) -> _SpectralBlocks:
    """Diagonalize C = S V_r* J V_r S and validate its numerical inertia."""
    values = svd.retained_values
    right_basis = svd.right_basis
    gram = right_basis.conj().T @ (signs[:, None] * right_basis)
    compressed = hermitian_part(values[:, None] * gram * values)
    eigenvalues, eigenvectors = np.linalg.eigh(compressed)
    threshold = float(spectral_tolerance * values[0] ** 2)
    positive = np.flatnonzero(eigenvalues > threshold)[::-1]
    negative = np.flatnonzero(eigenvalues < -threshold)
    zero = np.flatnonzero(np.abs(eigenvalues) <= threshold)
    l_plus, l_minus, neutral_count = len(positive), len(negative), len(zero)
    p0 = p - l_plus - neutral_count
    q0 = len(signs) - p - l_minus - neutral_count
    if min(p0, q0) < 0:
        raise NumericalBreakdownError(
            f"Numerical structure incompatible with J: r={svd.rank}, l+={l_plus}, "
            f"l-={l_minus}, j={neutral_count}, p0={p0}, q0={q0}. "
            "Inspect rank/spectral cutoffs.",
            stage="spectrum",
        )
    return _SpectralBlocks(
        eigenvalues=eigenvalues,
        positive_vectors=eigenvectors[:, positive],
        negative_vectors=eigenvectors[:, negative],
        zero_vectors=eigenvectors[:, zero],
        positive_values=np.sqrt(eigenvalues[positive]),
        negative_values=np.sqrt(-eigenvalues[negative]),
        threshold=threshold,
        structure=HSVDStructure(svd.rank, l_plus, l_minus, neutral_count, p0, q0),
    )


def _build_right_blocks(
    svd: _TruncatedSVD, spectrum: _SpectralBlocks, signs: RealArray
) -> _RightBasisBlocks:
    """Construct X+, X- and simultaneous normalized chains E, G for scaled A."""
    right_basis = svd.right_basis
    values = svd.retained_values[:, None]
    positive = (
        right_basis @ (values * spectrum.positive_vectors)
    ) / spectrum.positive_values
    negative = (
        right_basis @ (values * spectrum.negative_vectors)
    ) / spectrum.negative_values
    neutral = right_basis @ (values * spectrum.zero_vectors)
    preliminary_dual = signs[:, None] * (right_basis @ (spectrum.zero_vectors / values))
    dual_gram = hermitian_part(
        preliminary_dual.conj().T @ (signs[:, None] * preliminary_dual)
    )
    # The full Gram correction couples distinct chains; using only its
    # diagonal would leave nonzero J-products between different chains.
    dual = preliminary_dual - (neutral @ dual_gram) / 2
    return _RightBasisBlocks(positive, negative, neutral, dual)


def _orthogonal_complement(
    blocks: _RightBasisBlocks, signs: RealArray, basis_tolerance: float
) -> NumericArray:
    """Find ker(T* J) using an equilibrated T = [X+, X-, E, G]."""
    active_basis = np.column_stack(
        (blocks.positive, blocks.negative, blocks.neutral, blocks.dual)
    )
    column_scale = np.max(np.abs(active_basis), axis=0)
    if np.any(column_scale == 0):
        raise NumericalBreakdownError("A chain/basis column vanished", stage="basis")
    equilibrated = active_basis / column_scale
    equilibrated = equilibrated / np.linalg.norm(equilibrated, axis=0)
    _, singular_values, right_adjoint = np.linalg.svd(
        equilibrated.conj().T * signs, full_matrices=True
    )
    active_count = active_basis.shape[1]
    resolved_rank = np.count_nonzero(
        singular_values > basis_tolerance * singular_values[0]
    )
    if resolved_rank != active_count:
        raise NumericalBreakdownError(
            "The active right subspace is numerically rank deficient", stage="basis"
        )
    return right_adjoint.conj().T[:, active_count:]


def _complete_right_basis(
    blocks: _RightBasisBlocks,
    signs: RealArray,
    structure: HSVDStructure,
    basis_tolerance: float,
    check_tolerance: float,
) -> tuple[NumericArray, NumericArray]:
    """Complete the active right subspace with J-positive/negative vectors.

    Equilibrated E,G span the same space as V_v,V_w, but do not lose the
    smaller chain direction when those final columns nearly coincide.
    """
    columns = len(signs)
    complement_size = structure.p0 + structure.q0
    if complement_size == 0 and check_tolerance * np.sqrt(columns) < 1:
        # The mandatory final check gives ||V*JV-J||_2 <= ||V*JV-J||_F < 1,
        # certifying nonsingularity without an additional full-size SVD.
        empty = np.empty((columns, 0), dtype=blocks.positive.dtype)
        return empty, empty
    complement = _orthogonal_complement(blocks, signs, basis_tolerance)
    if complement.shape[1] != complement_size:
        raise NumericalBreakdownError("Unexpected complement dimension", stage="basis")
    if complement_size == 0:
        return complement, complement
    return _normalize_complement(complement, signs, structure, basis_tolerance)


def _normalize_complement(
    complement: NumericArray,
    signs: RealArray,
    structure: HSVDStructure,
    basis_tolerance: float,
) -> tuple[NumericArray, NumericArray]:
    """Resolve the complement inertia and scale its basis to J-norms +1/-1."""
    gram = hermitian_part(complement.conj().T @ (signs[:, None] * complement))
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    # ||W||_2 = ||J||_2 = 1 requires an absolute rounding scale. Scaling by
    # ||W*JW|| alone would overlook cancellation when that whole matrix is tiny.
    if np.any(np.abs(eigenvalues) <= basis_tolerance):
        raise NumericalBreakdownError(
            "The J-inertia of the complement is unresolved", stage="complement"
        )
    positive = np.flatnonzero(eigenvalues > 0)
    negative = np.flatnonzero(eigenvalues < 0)
    if (len(positive), len(negative)) != (structure.p0, structure.q0):
        raise NumericalBreakdownError("Complement inertia mismatch", stage="complement")
    return (
        (complement @ eigenvectors[:, positive]) / np.sqrt(eigenvalues[positive]),
        (complement @ eigenvectors[:, negative]) / np.sqrt(-eigenvalues[negative]),
    )


def _canonical_matrix(
    shape: tuple[int, int], p: int, spectrum: _SpectralBlocks, scale: float
) -> RealArray:
    """Place nonzero values and the two isotropic unit blocks into Sigma."""
    structure = spectrum.structure
    canonical = np.zeros(shape)
    positive = np.arange(structure.l_plus)
    negative = np.arange(structure.l_minus)
    neutral = np.arange(structure.j)
    canonical[positive, positive] = spectrum.positive_values * scale
    canonical[structure.l_plus + negative, p + negative] = (
        spectrum.negative_values * scale
    )
    canonical[structure.l + neutral, structure.l_plus + neutral] = 1
    canonical[structure.l + neutral, p + structure.l_minus + neutral] = 1
    return canonical


def _assemble_factors(
    svd: _TruncatedSVD,
    spectrum: _SpectralBlocks,
    blocks: _RightBasisBlocks,
    complement: tuple[NumericArray, NumericArray],
    scale: float,
    p: int,
) -> tuple[NumericArray, RealArray, NumericArray]:
    """Restore the input scale and assemble factors in canonical block order."""
    positive_complement, negative_complement = complement
    # Restoring A = scale*A0 requires E -> scale*E and G -> G/scale.
    # Scaling Sigma's isotropic unit blocks would destroy canonicity.
    neutral_half = (blocks.neutral * scale) / 2
    original_dual = divide_by_real(blocks.dual, scale)
    right_factor = np.column_stack((
        blocks.positive, neutral_half + original_dual, positive_complement,
        blocks.negative, neutral_half - original_dual, negative_complement,
    ))
    left_factor = np.column_stack((
        svd.left_basis @ spectrum.positive_vectors,
        svd.left_basis @ spectrum.negative_vectors,
        svd.left_basis @ spectrum.zero_vectors,
        svd.left_complement,
    ))
    shape = (left_factor.shape[0], right_factor.shape[0])
    return left_factor, _canonical_matrix(shape, p, spectrum, scale), right_factor


def _calculate_diagnostics(
    scaled_matrix: NumericArray,
    scale: float,
    left_factor: NumericArray,
    canonical: RealArray,
    right_factor: NumericArray,
    signs: RealArray,
    truncation_error: float,
) -> HSVDDiagnostics:
    """Measure all factor identities against the original scaled input."""
    rows, columns = scaled_matrix.shape
    left_product = left_factor @ (canonical / scale)
    reconstruction = left_product @ right_factor.conj().T
    transformed = ((scaled_matrix * signs) @ right_factor) * signs
    matrix_norm = frobenius_norm(scaled_matrix)
    product_norm = frobenius_norm(left_product)
    left_gram = euclidean_gram(left_factor)
    right_gram = right_factor.conj().T @ (signs[:, None] * right_factor)
    return HSVDDiagnostics(
        reconstruction_error=(
            frobenius_norm(scaled_matrix - reconstruction) / (matrix_norm or 1.0)
        ),
        unitarity_error=frobenius_norm(left_gram - np.eye(rows)) / np.sqrt(rows),
        j_unitarity_error=(
            frobenius_norm(right_gram - np.diag(signs)) / np.sqrt(columns)
        ),
        transform_error=(
            frobenius_norm(transformed - left_product) / (product_norm or 1.0)
        ),
        norm_v=float(np.linalg.norm(right_factor, 2)),
        rank_truncation_error=truncation_error,
    )


def _check_diagnostics(diagnostics: HSVDDiagnostics, check_tolerance: float) -> None:
    """Reject nonfinite diagnostics or a failed factor identity."""
    residuals = (
        diagnostics.reconstruction_error,
        diagnostics.unitarity_error,
        diagnostics.j_unitarity_error,
        diagnostics.transform_error,
    )
    all_metrics = (*residuals, diagnostics.norm_v, diagnostics.rank_truncation_error)
    if not np.isfinite(all_metrics).all():
        raise NumericalBreakdownError(
            "Factor diagnostics contain nonfinite values",
            stage="verification", diagnostics=diagnostics,
        )
    if max(residuals) > check_tolerance:
        raise NumericalBreakdownError(
            f"Factor residuals exceed check_tol={check_tolerance:g}: "
            f"A={residuals[0]:.3e}, U={residuals[1]:.3e}, J={residuals[2]:.3e}, "
            f"AJVJ={residuals[3]:.3e}. The numerical structure/scale is not resolved.",
            stage="verification", diagnostics=diagnostics,
        )
