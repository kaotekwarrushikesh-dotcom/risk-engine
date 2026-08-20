"""Phase 5: parametric (variance-covariance) VaR.

    Normal:     VaR(c) = -(mu + z_(1-c) * sigma)
    Student-t:  VaR(c) = -(mu + t_(1-c),v * sigma * sqrt((v - 2) / v))

Instead of reading a quantile off the data, this fits a distribution and reads the quantile
off the *model*. The trade is a real one in both directions: the model can produce a loss
worse than anything in the sample, which historical VaR (Phase 4) structurally cannot, but
it can only do so if the assumed shape is right.

**For equity returns the normal assumption is wrong, and wrong in the direction that
matters.** Daily returns are fat-tailed and left-skewed: extreme days happen far more often
than a normal distribution allows. A normal VaR therefore understates the tail, and it
understates it most at exactly the high confidence levels where VaR is supposed to be
informative. This module does not quietly assume normality: it runs a Jarque-Bera test on
the returns, reports the excess kurtosis, and warns when the assumption fails, which for
almost any real equity series it does.

**The Student-t variant is the honest default for fat tails**, with the degrees of freedom
estimated from the data rather than picked. The scaling factor `sqrt((v - 2) / v)` matters
and is easy to omit: a standard t distribution with v degrees of freedom has variance
`v / (v - 2)`, not 1, so using the raw t quantile against a sample standard deviation
double-counts the spread and inflates the VaR. The factor rescales the t to unit variance so
sigma means what it says.

**Where this and Phase 4 disagree is the finding, not a defect.** Comparing normal VaR to
historical VaR on the same series measures how much the normal assumption is costing: if the
market has genuinely had more bad days than a bell curve permits, the historical number will
be the larger one, and the size of that gap is a direct read on the fat tail.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252

# Excess kurtosis above this is a materially fat tail; the normal distribution has 0 by
# definition, and daily equity returns typically land between 2 and 8.
FAT_TAIL_KURTOSIS = 1.0
# Jarque-Bera below this p-value rejects normality at the usual 5% level.
NORMALITY_ALPHA = 0.05
# A t distribution needs more than 2 degrees of freedom for its variance to exist at all.
MIN_DEGREES_OF_FREEDOM = 2.5
MAX_DEGREES_OF_FREEDOM = 250.0


@dataclass(frozen=True)
class DistributionFit:
    """What the returns actually look like, before any distribution is assumed."""

    mean: float
    std: float
    skewness: float
    excess_kurtosis: float
    jarque_bera_stat: float
    jarque_bera_p: float
    n_observations: int

    @property
    def is_normal(self) -> bool:
        """Whether normality survives a Jarque-Bera test at the 5% level."""
        return self.jarque_bera_p >= NORMALITY_ALPHA

    @property
    def has_fat_tails(self) -> bool:
        return self.excess_kurtosis > FAT_TAIL_KURTOSIS

    def verdict(self) -> str:
        if self.is_normal:
            return (
                f"Normality is not rejected (Jarque-Bera p = {self.jarque_bera_p:.3f}), so a "
                "normal VaR is defensible on this series. That is unusual for daily equity "
                "returns and is worth checking rather than relying on."
            )
        return (
            f"Normality is rejected (Jarque-Bera p = {self.jarque_bera_p:.2g}, excess kurtosis "
            f"{self.excess_kurtosis:.2f} against 0 for a normal). Extreme days happen more often "
            "than a bell curve allows, so a normal VaR understates the tail, and understates it "
            "most at the high confidence levels where VaR is meant to be informative."
        )


@dataclass(frozen=True)
class ParametricVaRResult:
    """One parametric VaR estimate and the assumption it rests on."""

    method: str
    confidence: float
    horizon_days: int
    var_return: float
    var_value: float | None
    fit: DistributionFit
    degrees_of_freedom: float | None
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def var_pct(self) -> float:
        return self.var_return * 100.0

    def summary(self) -> str:
        horizon = "1 day" if self.horizon_days == 1 else f"{self.horizon_days} days"
        line = f"{self.confidence:.0%} {horizon} {self.method} VaR: {self.var_pct:.2f}%"
        if self.var_value is not None:
            line += f" ({self.var_value:,.0f})"
        return line


def fit_distribution(returns: pd.Series) -> DistributionFit:
    """Describe the return distribution and test it against normality.

    This runs before any parametric VaR so the assumption is checked rather than asserted.
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 8:
        raise ValueError("at least eight observations are needed to test distribution shape")

    values = clean.to_numpy(dtype=float)
    jb_stat, jb_p = stats.jarque_bera(values)

    return DistributionFit(
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1)),
        skewness=float(stats.skew(values)),
        excess_kurtosis=float(stats.kurtosis(values)),  # Fisher: normal is 0
        jarque_bera_stat=float(jb_stat),
        jarque_bera_p=float(jb_p),
        n_observations=n,
    )


def estimate_degrees_of_freedom(returns: pd.Series) -> float:
    """Fit a Student-t to the returns and return its degrees of freedom.

    Estimated rather than chosen: the whole point of the t variant is to let the data say how
    fat the tail is. Bounded on both sides, since a fitted `v` at or below 2 implies infinite
    variance (the distribution has no standard deviation to scale) and a very large `v` is
    indistinguishable from a normal, at which point the t adds machinery and no information.
    """
    clean = returns.dropna().to_numpy(dtype=float)
    df, _loc, _scale = stats.t.fit(clean)
    return float(np.clip(df, MIN_DEGREES_OF_FREEDOM, MAX_DEGREES_OF_FREEDOM))


def _to_simple_loss(log_return: float) -> float:
    """A log return converted to a positive simple-return loss. See var_historical."""
    return float(-(np.exp(log_return) - 1.0))


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
    distribution: str = "normal",
) -> ParametricVaRResult:
    """VaR from a fitted distribution rather than from the empirical quantile.

    `distribution` is "normal" or "t". The t variant estimates its degrees of freedom from
    the data and rescales to unit variance so that sigma keeps its meaning.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0.5 and 1, got {confidence}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be at least 1, got {horizon_days}")
    if distribution not in ("normal", "t"):
        raise ValueError(f"unknown distribution: {distribution!r}, expected 'normal' or 't'")

    fit = fit_distribution(returns)
    warnings: list[str] = []
    notes: list[str] = []
    tail = 1.0 - confidence

    # Mean and volatility both scale with the horizon, but differently: the mean grows
    # linearly in time while the standard deviation grows with its square root. Scaling both
    # by sqrt(h) is a common and quietly wrong shortcut.
    mu = fit.mean * horizon_days
    sigma = fit.std * np.sqrt(horizon_days)

    if distribution == "normal":
        z = float(stats.norm.ppf(tail))
        quantile = mu + z * sigma
        degrees_of_freedom = None
        notes.append(
            f"Normal quantile z = {z:.3f} at {confidence:.0%}. Mean scaled linearly with the "
            "horizon and volatility by its square root, since they scale differently."
        )
        if not fit.is_normal:
            warnings.append(
                f"the normal assumption behind this number is rejected on this series "
                f"(Jarque-Bera p = {fit.jarque_bera_p:.2g}, excess kurtosis "
                f"{fit.excess_kurtosis:.2f}), so this VaR understates the tail. Compare it "
                "against the historical and Student-t figures rather than using it alone."
            )
    else:
        degrees_of_freedom = estimate_degrees_of_freedom(returns)
        t_quantile = float(stats.t.ppf(tail, degrees_of_freedom))
        # A standard t with v degrees of freedom has variance v/(v-2), not 1. Without this
        # rescaling the spread is counted twice and the VaR comes out inflated.
        unit_variance_scale = np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
        quantile = mu + t_quantile * unit_variance_scale * sigma
        notes.append(
            f"Student-t with {degrees_of_freedom:.1f} degrees of freedom, estimated from the "
            f"data. Quantile {t_quantile:.3f} rescaled by "
            f"sqrt((v-2)/v) = {unit_variance_scale:.3f} so the distribution has unit variance "
            "and sigma keeps its meaning."
        )
        if degrees_of_freedom >= MAX_DEGREES_OF_FREEDOM:
            warnings.append(
                f"the fitted degrees of freedom hit the {MAX_DEGREES_OF_FREEDOM:.0f} ceiling, "
                "meaning the t is indistinguishable from a normal here and is adding machinery "
                "rather than information."
            )

    var_return = _to_simple_loss(quantile)

    if horizon_days > 1:
        notes.append(
            f"Scaled to {horizon_days} days assuming returns are independent day to day. "
            "Volatility clusters in practice, so a stressed multi-day loss is worse than this."
        )

    return ParametricVaRResult(
        method="parametric-normal" if distribution == "normal" else "parametric-t",
        confidence=confidence,
        horizon_days=horizon_days,
        var_return=var_return,
        var_value=var_return * portfolio_value if portfolio_value is not None else None,
        fit=fit,
        degrees_of_freedom=degrees_of_freedom,
        warnings=tuple(warnings),
        notes=tuple(notes),
    )


def normal_vs_historical_gap(parametric_var_return: float, historical_var_return: float) -> dict:
    """How much the normal assumption is costing, measured rather than asserted.

    A positive `gap` means the empirical tail is worse than the normal model allows, which is
    the expected direction for equities and a direct read on the fat tail. A negative gap is
    worth investigating rather than celebrating: it usually means the sample simply has not
    contained a bad enough day yet, which is Phase 4's structural blind spot rather than
    evidence that the normal model is adequate.
    """
    gap = historical_var_return - parametric_var_return
    ratio = (historical_var_return / parametric_var_return
             if parametric_var_return > 0 else float("nan"))
    if gap > 0:
        reading = (
            f"The empirical tail is {gap:.2%} worse than the normal model allows "
            f"({ratio:.2f}x), which is the fat tail showing up as a number."
        )
    else:
        reading = (
            f"The normal model is the more conservative of the two here by {-gap:.2%}. That "
            "usually means the sample has not yet contained a loss bad enough to move the "
            "empirical quantile, not that returns are well behaved."
        )
    return {"gap": gap, "ratio": ratio, "reading": reading}
