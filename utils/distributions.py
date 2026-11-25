"""Small containers for the different bits of a distribution.

The aim is to keep these entirely data-focused and small so they can be saved and used downstream.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, repr=False)
class BaseDistribution:
    """Container for a distribution."""

    mean: np.ndarray
    cov: np.ndarray


@dataclass(frozen=True, repr=False)
class Posterior(BaseDistribution):
    """Container for a posterior distribution."""

    d: np.ndarray  # data on which the posterior is conditioned on
    ln_Z: float


@dataclass(frozen=True, repr=True)
class PosteriorPredictive(BaseDistribution):
    """Container for a posterior predictive distribution.

    Currently don't have the normalisation factor for the ppd, although it's probably available analytically.
    """

    d: np.ndarray  # data on which the ppd is conditioned on
