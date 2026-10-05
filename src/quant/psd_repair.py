"""Make an inconsistent correlation matrix usable, and say how far it moved.

RESEARCH_ONLY. Pure numerics: no market, no stake, no wager.

THE PROBLEM. ``parlay.correlation_matrix`` assembles a matrix from PAIRWISE
estimates, each fitted on its own bucket of games. Nothing makes a set of
pairwise numbers mutually consistent, and an inconsistent set describes no joint
distribution — so ``_validated_cholesky`` raises and the ticket cannot be priced
at all. An entirely ordinary case:

    A correlates +0.75 with B and +0.75 with C, and B correlates -0.40 with C

        eigenvalues  [-0.2794, 1.4000, 1.8794]      cholesky: FAILS

That is not a typo in the inputs. Three legs can each be individually plausible
and jointly impossible, and the existing error message asks a human to "check
the pairwise values against each other rather than adjusting one in isolation"
— which is this computation, done by hand.

WHAT THIS DOES, AND WHY IT IS NOT A SILENT FIX. ``correlation_matrix``'s
docstring is right that quietly repairing would "invent dependence nobody
supplied". So the repair is:

  * OFF by default. Callers opt in.
  * REPORTED. ``RepairResult`` carries the minimum eigenvalue before, the
    largest single entry that moved, and whether anything moved at all.
  * BOUNDED. Above ``max_shift`` the repair ABSTAINS rather than returning a
    matrix that no longer resembles what was measured. A projection that has to
    move an entry by 0.3 is not cleaning up floating-point noise; it is telling
    you two of the estimates genuinely contradict each other.

Method: eigendecompose the symmetric part, clip eigenvalues up to ``eps``,
reconstruct, then rescale to a unit diagonal — the standard nearest-PSD
projection. Optional shrinkage toward the identity runs FIRST, because pulling
every off-diagonal toward zero is a weaker claim than reshaping the matrix and
often removes the need to reshape at all.

``eps`` defaults above zero on purpose. Clipping to exactly 0 leaves the matrix
positive SEMI-definite, and ``numpy.linalg.cholesky`` wants positive DEFINITE;
a boundary matrix can still fail on the next floating-point operation.

Adapted from the eigenvalue-clipping and identity-shrinkage approach in
``EdgarParra565/player-performance-forecaster``'s ``correlation_calibration.py``
— see ``docs/external_repo_review_2026-10.md`` §2.1. The reporting, the
abstention and the default-off are this project's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Smallest eigenvalue the repaired matrix is allowed to have. Above 0 so the
#: result is positive DEFINITE and survives a Cholesky.
DEFAULT_EPS = 1e-8
#: Off-diagonal magnitude cap. A |rho| of exactly 1 makes two legs the same leg.
DEFAULT_MAX_ABS = 0.999
#: Refuse rather than repair past this much movement in any single entry.
DEFAULT_MAX_SHIFT = 0.15


class PsdRepairError(ValueError):
    """The matrix cannot be made usable without changing what it says."""


@dataclass(frozen=True)
class RepairResult:
    """The usable matrix, and the full account of what was done to get it."""

    matrix: np.ndarray
    was_psd: bool
    min_eigenvalue_before: float
    min_eigenvalue_after: float
    max_entry_shift: float
    shrinkage: float = 0.0

    @property
    def repaired(self) -> bool:
        return not self.was_psd or self.shrinkage > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "was_psd": self.was_psd,
            "repaired": self.repaired,
            "min_eigenvalue_before": round(float(self.min_eigenvalue_before), 8),
            "min_eigenvalue_after": round(float(self.min_eigenvalue_after), 8),
            "max_entry_shift": round(float(self.max_entry_shift), 6),
            "shrinkage": float(self.shrinkage),
        }


def is_psd(matrix: np.ndarray, *, eps: float = 0.0) -> bool:
    """True when every eigenvalue is at least ``eps``."""
    array = np.asarray(matrix, dtype=float)
    if array.size == 0:
        return True
    return bool(np.linalg.eigvalsh((array + array.T) / 2.0).min() >= eps)


def nearest_correlation(
    matrix: np.ndarray,
    *,
    shrinkage: float = 0.0,
    eps: float = DEFAULT_EPS,
    max_abs: float = DEFAULT_MAX_ABS,
    max_shift: float | None = DEFAULT_MAX_SHIFT,
) -> RepairResult:
    """
    The nearest valid correlation matrix, with the distance reported.

    ``shrinkage`` in [0, 1] blends toward the identity before any reshaping:
    ``(1 - w) * R + w * I``. It is a weaker, more honest adjustment than
    eigenvalue surgery — it says "trust these estimates less" rather than
    "these estimates were different numbers".

    COUNTER-INTUITIVE AND DELIBERATE: shrinkage usually INCREASES
    ``max_entry_shift``, because the shift is measured against what was
    originally supplied and shrinkage itself moves every off-diagonal toward
    zero. On one mildly inconsistent matrix, projection alone moved 0.0065
    while ``shrinkage=0.10`` moved 0.0600. That is the right accounting — the
    caller asked for less trust and is told the full distance from the
    measurement — but it means shrinkage can trip ``max_shift`` on a matrix
    that projection alone would have repaired.

    Raises ``PsdRepairError`` when the repair would move a single entry by more
    than ``max_shift``. Pass ``max_shift=None`` to allow any movement, which a
    caller should only do if it is reporting the figure itself.
    """
    original = np.asarray(matrix, dtype=float)
    if original.ndim != 2 or original.shape[0] != original.shape[1]:
        raise PsdRepairError(f"not a square matrix: shape {original.shape}")
    size = original.shape[0]
    if size == 0:
        return RepairResult(original.copy(), True, 0.0, 0.0, 0.0, 0.0)
    if not np.all(np.isfinite(original)):
        raise PsdRepairError(
            "the matrix contains a non-finite entry, which is a missing "
            "estimate rather than a correlation of any size"
        )

    weight = float(shrinkage)
    if not 0.0 <= weight <= 1.0:
        raise PsdRepairError(f"shrinkage must be in [0, 1], got {shrinkage!r}")

    symmetric = (original + original.T) / 2.0
    min_before = float(np.linalg.eigvalsh(symmetric).min())
    already = min_before >= eps and weight == 0.0
    if already:
        out = symmetric.copy()
        np.fill_diagonal(out, 1.0)
        return RepairResult(out, True, min_before, min_before, 0.0, 0.0)

    identity = np.eye(size, dtype=float)
    work = (1.0 - weight) * symmetric + weight * identity if weight else symmetric

    # Eigenvalue clipping, then rescale so the diagonal is 1 again: scaling a
    # PSD matrix by a positive diagonal on both sides keeps it PSD.
    eigenvalues, eigenvectors = np.linalg.eigh(work)
    clipped = np.clip(eigenvalues, eps, None)
    rebuilt = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    scale = np.sqrt(np.clip(np.diag(rebuilt), eps, None))
    corr = rebuilt / np.outer(scale, scale)

    cap = abs(float(max_abs))
    np.clip(corr, -cap, cap, out=corr)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    off_diagonal = ~np.eye(size, dtype=bool)
    shift = float(np.abs(corr - symmetric)[off_diagonal].max()) if size > 1 else 0.0
    min_after = float(np.linalg.eigvalsh(corr).min())

    if max_shift is not None and shift > float(max_shift):
        raise PsdRepairError(
            f"repairing this matrix moves an entry by {shift:.4f}, past the "
            f"{float(max_shift):.4f} allowed. The pairwise estimates contradict "
            f"each other (minimum eigenvalue {min_before:.4f}) by more than a "
            "projection should paper over — refit the pairs or drop a leg "
            "rather than price this as it stands."
        )

    logger.info(
        "Correlation repair: min eigenvalue %.6f -> %.6f, largest entry moved "
        "%.4f%s",
        min_before, min_after, shift,
        f", shrinkage {weight:.2f}" if weight else "",
    )
    return RepairResult(
        matrix=corr,
        was_psd=min_before >= eps,
        min_eigenvalue_before=min_before,
        min_eigenvalue_after=min_after,
        max_entry_shift=shift,
        shrinkage=weight,
    )
