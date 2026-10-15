"""Hyperbolic SVD in the canonical form A = U Sigma V*, V* J V = J."""

from .core import (
    HSVDDiagnostics,
    HSVDResult,
    HSVDStructure,
    NumericalBreakdownError,
    hsvd,
)

__all__ = [
    "hsvd", "HSVDResult", "HSVDStructure", "HSVDDiagnostics",
    "NumericalBreakdownError",
]
