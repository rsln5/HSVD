"""Shared double-precision operations used by the HSVD implementations."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

RealArray: TypeAlias = NDArray[np.float64]
NumericArray: TypeAlias = RealArray | NDArray[np.complex128]


def component_scale(array: NumericArray) -> float:
    """Return the largest absolute real/imaginary component, or zero if empty."""
    if array.size == 0:
        return 0.0
    scale = float(np.max(np.abs(array.real)))
    if np.iscomplexobj(array):
        scale = max(scale, float(np.max(np.abs(array.imag))))
    return scale


def divide_by_real(array: NumericArray, divisor: float) -> NumericArray:
    """Divide by a nonzero real scalar without forming its reciprocal."""
    # Complex division may form 1/divisor internally and overflow when the
    # divisor is subnormal, even if the componentwise quotients are finite.
    if np.iscomplexobj(array):
        return array.real / divisor + 1j * (array.imag / divisor)
    return array / divisor


def frobenius_norm(array: NumericArray) -> float:
    """Compute a Frobenius/vector norm without squaring unscaled entries."""
    scale = component_scale(array)
    if scale == 0:
        return 0.0
    scaled = divide_by_real(array, scale)
    return float(scale * np.sqrt(np.sum(scaled.real**2) + np.sum(scaled.imag**2)))


def hermitian_part(matrix: NumericArray) -> NumericArray:
    """Symmetrize without overflowing large sums or erasing subnormal entries."""
    adjoint = matrix.conj().T
    limit = np.finfo(np.float64).max / 2
    if component_scale(matrix) <= limit:
        # Add first: halving each subnormal entry separately can erase or
        # double a nonzero entry even when the input is already Hermitian.
        return (matrix + adjoint) / 2
    result = np.empty_like(matrix)
    components = [(matrix.real, adjoint.real, result.real)]
    if np.iscomplexobj(matrix):
        components.append((matrix.imag, adjoint.imag, result.imag))
    for left, right, target in components:
        safe_sum = (np.abs(left) <= limit) & (np.abs(right) <= limit)
        target[safe_sum] = (left[safe_sum] + right[safe_sum]) / 2
        target[~safe_sum] = left[~safe_sum] / 2 + right[~safe_sum] / 2
    return result


def euclidean_gram(matrix: NumericArray) -> NumericArray:
    """Compute matrix* matrix, with a fallback for spurious BLAS FP errors."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            return matrix.conj().T @ matrix
    except FloatingPointError:
        # Some Accelerate BLAS versions signal spurious FP exceptions here.
        # Recompute only this contraction; final residual checks still apply.
        return np.einsum("ki,kj->ij", matrix.conj(), matrix, optimize=False)


def signature_signs(column_count: int, positive_count: int) -> RealArray:
    """Return the diagonal of J for 0 <= positive_count <= column_count."""
    signs = np.ones(column_count)
    signs[positive_count:] = -1
    return signs
