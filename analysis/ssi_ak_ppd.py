"""Analysis of the posterior predictive around SSI-AK paths."""
import numpy as np

from utils.distributions import Posterior, PosteriorPredictive

from ._types import dict_s


def _mahalanobis(d: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Compute the squared Mahalanobis distance.

    Yes, I know scipy has the mahalanobis distance in it.
    """
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular or not numerically positive definite")
    r = d - mean
    y = np.linalg.solve(L, r)
    return float(y @ y)


def ppd_mahalanobis_total(
    posterior: Posterior, ppd: PosteriorPredictive, context: dict_s
) -> dict_s:
    """AnalysisFn to compute the total Mahalanobis distance of the observed data from the ppd mean."""

    return {"mahalanobis_total": _mahalanobis(ppd.d, ppd.mean, ppd.cov)}


def ppd_mahalanobis_ssi_ak(
    posterior: Posterior, ppd: PosteriorPredictive, context: dict_s
) -> dict_s:
    """AnalysisFn to compute the total Mahalanobis distance of the observed data from the ppd mean for the SSI-AK subset."""
    filt = context["filt"]
    d = ppd.d[np.ix_(filt)]
    m = ppd.mean[np.ix_(filt)]
    cov = ppd.cov[np.ix_(filt, filt)]

    return {"mahalanobis_ssi_ak": _mahalanobis(d, m, cov)}


def ppd_mahalanobis_ssi_ak_complement(
    posterior: Posterior, ppd: PosteriorPredictive, context: dict_s
) -> dict_s:
    """AnalysisFn to compute the total Mahalanobis distance of the observed data from the ppd mean for complement of the SSI-AK subset."""
    filt = np.logical_not(context["filt"])
    d = ppd.d[np.ix_(filt)]
    m = ppd.mean[np.ix_(filt)]
    cov = ppd.cov[np.ix_(filt, filt)]

    return {"mahalanobis_ssi_ak_comp": _mahalanobis(d, m, cov)}
