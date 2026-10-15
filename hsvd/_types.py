"""Public result records and exceptions for the full HSVD interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._numeric import NumericArray, RealArray, signature_signs


@dataclass(frozen=True)
class HSVDStructure:
    """Numerical canonical structure; rank = l_plus + l_minus + j."""

    rank: int
    l_plus: int
    l_minus: int
    j: int
    p0: int
    q0: int

    @property
    def l(self) -> int:
        """Number of nonzero hyperbolic singular values."""
        return self.l_plus + self.l_minus


@dataclass(frozen=True)
class HSVDDiagnostics:
    """Errors measured on the actual returned factors, including truncation.

    reconstruction_error = ||A - U Sigma V*||_F / ||A||_F.
    unitarity_error = ||U* U - I||_F / sqrt(m).
    j_unitarity_error = ||V* J V - J||_F / sqrt(n).
    transform_error = ||A J V J - U Sigma||_F / ||U Sigma||_F.
    A zero denominator means an absolute error. norm_v is ||V||_2;
    rank_truncation_error is the relative discarded SVD tail.
    """

    reconstruction_error: float
    unitarity_error: float
    j_unitarity_error: float
    transform_error: float
    norm_v: float
    rank_truncation_error: float


class NumericalBreakdownError(ArithmeticError):
    """The chosen numerical structure or factors could not be validated."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        diagnostics: HSVDDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class HSVDResult:
    """Full factors for J = diag(I_p, -I_(n-p)).

    U, Sigma, V have shapes (m,m), (m,n), (n,n). U and V are real for
    real input, complex for complex input; Sigma is real. The spectrum and
    cutoffs refer to A/input_scale, not the original A. The container is
    frozen, but its NumPy arrays remain mutable.
    """

    U: NumericArray
    Sigma: RealArray
    V: NumericArray
    p: int
    structure: HSVDStructure
    diagnostics: HSVDDiagnostics
    input_scale: float
    singular_values_scaled: RealArray
    eigenvalues_scaled: RealArray
    rank_threshold_scaled: float
    spectral_threshold_scaled: float

    @property
    def J(self) -> RealArray:
        """Signature matrix of shape (n,n)."""
        return np.diag(signature_signs(self.V.shape[0], self.p))

    def reconstruct(self) -> NumericArray:
        """Evaluate U Sigma V* using the input scale to limit overflow.

        Raises FloatingPointError if the reconstruction is not representable.
        """
        if self.input_scale == 0:
            return np.zeros(self.Sigma.shape, dtype=self.U.dtype)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scaled_product = self.U @ (self.Sigma / self.input_scale)
            return (scaled_product @ self.V.conj().T) * self.input_scale
