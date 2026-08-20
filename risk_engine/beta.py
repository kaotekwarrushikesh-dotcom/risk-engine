"""Phase 3a: beta against a configurable benchmark.

Two equivalent routes to the same number, both implemented so they can be cross-checked
against each other:

    Beta = Cov(asset, market) / Var(market)                    (moments)
    Beta = slope of OLS regression of asset returns on market returns  (regression)

They agree exactly for simple OLS (the covariance formula *is* the closed-form solution to
the regression), so computing both and asserting they match is a cheap correctness check
that catches an indexing or alignment bug immediately rather than after a bad beta has
already fed into a VaR or CAPM figure downstream. The regression route additionally
produces R-squared and a t-statistic, which the moments formula does not, so it is the one
carried forward for display.

**R-squared says how much of the beta is worth trusting.** A beta of 1.4 with R-squared of
0.05 means the market explains almost none of this asset's return variation, and the 1.4 is
mostly noise wearing the shape of a real number. A beta of 1.4 with R-squared of 0.6 is a
different, much more trustworthy claim, even though the headline number is identical.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class BetaResult:
    """Beta, computed two independent ways, plus the statistics needed to judge it."""

    beta_moments: float
    beta_regression: float
    alpha: float
    r_squared: float
    correlation: float
    standard_error: float
    t_statistic: float
    p_value: float
    n_observations: int
    benchmark_name: str
    window_description: str

    def is_significant(self, level: float = 0.05) -> bool:
        """Whether beta is statistically distinguishable from zero at the given level."""
        return self.p_value < level

    @property
    def confidence(self) -> str:
        if self.n_observations < 30:
            return "low"
        if self.r_squared >= 0.30:
            return "high"
        if self.r_squared >= 0.10:
            return "medium"
        return "low"


def align_returns(asset: pd.Series, benchmark: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Line up two return series on their common dates.

    An unaligned regression (different lengths, mismatched dates silently zipped together
    positionally) produces a number that looks like a beta and is not one, so the join is
    explicit and inner.
    """
    joined = pd.concat([asset.rename("asset"), benchmark.rename("benchmark")], axis=1).dropna()
    return joined["asset"], joined["benchmark"]


def compute_beta(
    asset_returns: pd.Series, benchmark_returns: pd.Series,
    benchmark_name: str = "benchmark", window_description: str = "",
) -> BetaResult:
    """Beta by both the covariance formula and OLS regression, cross-checked against each other."""
    asset, bench = align_returns(asset_returns, benchmark_returns)
    n = len(asset)

    if n < 2 or bench.var(ddof=1) == 0:
        return BetaResult(
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), n, benchmark_name, window_description,
        )

    cov_matrix = np.cov(asset, bench, ddof=1)
    beta_moments = float(cov_matrix[0, 1] / cov_matrix[1, 1])

    slope, intercept, r_value, p_value, std_err = stats.linregress(bench, asset)

    # The two routes must agree: they are the same estimator by construction, so a
    # meaningful gap between them means an alignment bug, not a modelling choice.
    if not np.isclose(beta_moments, slope, rtol=1e-6, atol=1e-9):
        raise AssertionError(
            f"beta via covariance ({beta_moments:.6f}) and via regression ({slope:.6f}) "
            "disagree; this should be mathematically impossible for simple OLS and points "
            "at an alignment bug, not a modelling difference"
        )

    t_stat = slope / std_err if std_err > 0 else float("nan")

    return BetaResult(
        beta_moments=beta_moments, beta_regression=float(slope), alpha=float(intercept),
        r_squared=float(r_value**2), correlation=float(r_value), standard_error=float(std_err),
        t_statistic=float(t_stat), p_value=float(p_value), n_observations=n,
        benchmark_name=benchmark_name, window_description=window_description,
    )


def rolling_beta(
    asset_returns: pd.Series, benchmark_returns: pd.Series, window: int,
) -> pd.Series:
    """Beta recomputed on a rolling window, to see whether a stock's market sensitivity is
    stable over time or has been drifting."""
    asset, bench = align_returns(asset_returns, benchmark_returns)
    cov = asset.rolling(window).cov(bench)
    var = bench.rolling(window).var()
    return (cov / var).rename("rolling_beta")
