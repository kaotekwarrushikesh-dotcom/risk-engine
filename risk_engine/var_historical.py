"""Phase 4: historical Value at Risk.

    VaR(c) = the loss that is exceeded only (1 - c) of the time

Historical VaR reads that number straight off the empirical distribution: sort the observed
returns, take the (1 - c) quantile, report it as a positive loss. It assumes nothing about
the shape of the distribution, which is its whole advantage over the parametric approach in
Phase 5. Fat tails, skew and the occasional violent day are all already in the data, so
nothing has to be assumed about them.

**What it cannot do, and this is the important part.** Historical VaR is bounded below by
the worst loss in the sample. It has no mechanism for producing a number worse than
something that has already happened, so on a quiet sample it will confidently report a
small VaR precisely because nothing bad has occurred yet. That is not a flaw to be
engineered around; it is what the method is. This module reports the worst observed loss
alongside every VaR figure so the reader can see how close the estimate sits to the edge of
the evidence, and flags the case where the two are nearly the same.

**A tail estimate rests on very few observations, and the count is reported.** A 99% VaR on
250 trading days is a quantile supported by about 2.5 observations. The point estimate looks
just as precise as any other number on the screen, so `tail_observations` and a bootstrap
confidence interval travel with it: a 99% VaR of 3.1% with an interval spanning 2.4% to 4.6%
is a different statement from the same 3.1% with a narrow one, and the difference should not
be something the reader has to infer.

**Sign convention.** VaR is returned as a **positive number meaning a loss**. A VaR of 0.028
means "a 2.8% loss". Losses are not returned as negative numbers here, because a risk report
that mixes positive and negative loss conventions is how a sign error reaches a decision.

**Log returns are converted before being reported.** The engine works in log returns because
they scale across time (see `returns.py`), but a log return is not a percentage loss: a
-5% log return is a -4.88% actual loss. The conversion `exp(r) - 1` is applied to the
quantile before it is reported, which matters little at one day and materially at ten.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 20250820

# Below this many observations beyond the quantile, the estimate rests on so few points that
# it should be read as indicative. At 99% confidence this is reached at roughly 500
# observations, which is about two years of daily data.
MIN_TAIL_OBSERVATIONS = 5

# When the VaR estimate sits within this fraction of the worst loss ever observed, the
# method is effectively reporting its own sample maximum and has no headroom left.
AT_THE_EDGE_RATIO = 0.9


@dataclass(frozen=True)
class VaRResult:
    """One VaR estimate, with everything needed to judge how much to trust it."""

    method: str
    confidence: float
    horizon_days: int
    var_return: float
    var_value: float | None
    n_observations: int
    tail_observations: float
    worst_observed_loss: float
    ci_low: float
    ci_high: float
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def var_pct(self) -> float:
        return self.var_return * 100.0

    @property
    def at_the_edge_of_the_data(self) -> bool:
        """Whether the estimate is effectively reporting the worst thing that ever happened."""
        if np.isnan(self.worst_observed_loss) or self.worst_observed_loss <= 0:
            return False
        return self.var_return >= self.worst_observed_loss * AT_THE_EDGE_RATIO

    def summary(self) -> str:
        horizon = "1 day" if self.horizon_days == 1 else f"{self.horizon_days} days"
        line = (f"{self.confidence:.0%} {horizon} {self.method} VaR: {self.var_pct:.2f}%")
        if self.var_value is not None:
            line += f" ({self.var_value:,.0f})"
        return line


def _to_simple_loss(log_return_quantile: float) -> float:
    """Convert a log-return quantile into a positive simple-return loss.

    A -5% log return is a 4.88% loss, not a 5% one. The gap is immaterial at one day and
    material at ten, and reporting the log figure as if it were a percentage loss would
    overstate the loss slightly but consistently, in the direction that looks conservative
    and is simply wrong.
    """
    return float(-(np.exp(log_return_quantile) - 1.0))


def scale_to_horizon(daily_log_sigma_quantile: float, horizon_days: int) -> float:
    """Square-root-of-time scaling of a daily log-return quantile.

    Valid only if returns are independent and identically distributed, which they are not:
    volatility clusters, so a ten-day loss in a stressed market is worse than sqrt(10) times
    a calm one-day loss. This is the standard regulatory approximation (Basel uses exactly
    this scaling) and is used here for comparability, with the assumption stated rather than
    buried. Phase 8's GARCH model is what addresses the clustering this cannot.
    """
    return daily_log_sigma_quantile * np.sqrt(horizon_days)


def _bootstrap_interval(
    returns: np.ndarray, confidence: float, samples: int, seed: int,
) -> tuple[float, float]:
    """A 95% confidence interval for the VaR estimate itself, by resampling.

    The interval answers "how much would this number move if the same process had produced a
    different sample of the same size", which is the relevant uncertainty for a quantile
    estimated from a few hundred observations. It is not a claim about future losses.
    """
    n = len(returns)
    if n < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    quantile_level = 1.0 - confidence
    draws = rng.choice(returns, size=(samples, n), replace=True)
    quantiles = np.quantile(draws, quantile_level, axis=1)
    losses = -(np.exp(quantiles) - 1.0)
    return float(np.percentile(losses, 2.5)), float(np.percentile(losses, 97.5))


def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> VaRResult:
    """VaR from the empirical quantile of observed log returns.

    `returns` are log returns (the engine's default). `confidence` is the confidence level,
    so 0.95 means the loss exceeded on 5% of days. The result is a positive loss fraction.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0.5 and 1, got {confidence}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be at least 1, got {horizon_days}")

    clean = returns.dropna()
    n = len(clean)
    warnings: list[str] = []
    notes: list[str] = []

    if n < 2:
        raise ValueError("at least two return observations are needed for a VaR estimate")

    values = clean.to_numpy(dtype=float)
    quantile_level = 1.0 - confidence

    # Linear interpolation between order statistics is the default and is kept deliberately.
    # The alternative ("lower") would always land on an actually-observed return, which
    # sounds more honest but systematically overstates the loss at high confidence by
    # rounding to the worse of the two neighbouring observations.
    daily_quantile = float(np.quantile(values, quantile_level))
    scaled_quantile = scale_to_horizon(daily_quantile, horizon_days)
    var_return = _to_simple_loss(scaled_quantile)

    worst_log = float(values.min())
    worst_observed_loss = _to_simple_loss(scale_to_horizon(worst_log, horizon_days))

    # How many observations actually sit beyond the quantile. This is the sample size the
    # estimate genuinely rests on, and it is much smaller than `n`.
    tail_observations = float((values <= daily_quantile).sum())

    ci_low, ci_high = _bootstrap_interval(values, confidence, bootstrap_samples, seed)
    if horizon_days > 1:
        # The interval is computed on the daily quantile, so scale it the same way the point
        # estimate was scaled rather than re-running the bootstrap on scaled data.
        factor = np.sqrt(horizon_days)
        ci_low = _to_simple_loss(np.log1p(-ci_low) * factor)
        ci_high = _to_simple_loss(np.log1p(-ci_high) * factor)

    if tail_observations < MIN_TAIL_OBSERVATIONS:
        warnings.append(
            f"only {tail_observations:.0f} observations sit beyond the {confidence:.0%} "
            f"quantile, so this estimate rests on {tail_observations:.0f} data points rather "
            f"than {n}. It is indicative rather than measured."
        )

    at_the_edge = (
        worst_observed_loss > 0
        and not np.isnan(worst_observed_loss)
        and var_return >= worst_observed_loss * AT_THE_EDGE_RATIO
    )
    if at_the_edge:
        warnings.append(
            f"the estimate of {var_return:.2%} is within {AT_THE_EDGE_RATIO:.0%} of the worst "
            f"loss in the whole sample ({worst_observed_loss:.2%}). Historical VaR cannot "
            "produce a number worse than what has already happened, so at this confidence "
            "level on this sample it is reporting the edge of the data rather than measuring "
            "a tail."
        )

    if horizon_days > 1:
        notes.append(
            f"Scaled from one day to {horizon_days} by sqrt({horizon_days}), which assumes "
            "returns are independent day to day. Volatility clusters in practice, so a "
            "stressed multi-day loss is worse than this scaling implies."
        )

    notes.append(
        f"Read off {n} observations, {tail_observations:.0f} of which lie beyond the "
        f"quantile. Worst single loss in the sample: {worst_observed_loss:.2%}."
    )

    return VaRResult(
        method="historical",
        confidence=confidence,
        horizon_days=horizon_days,
        var_return=var_return,
        var_value=var_return * portfolio_value if portfolio_value is not None else None,
        n_observations=n,
        tail_observations=tail_observations,
        worst_observed_loss=worst_observed_loss,
        ci_low=ci_low,
        ci_high=ci_high,
        warnings=tuple(warnings),
        notes=tuple(notes),
    )


def rolling_historical_var(
    returns: pd.Series, window: int = 252, confidence: float = 0.95,
) -> pd.Series:
    """VaR recomputed on a rolling window, for charting how tail risk has moved.

    Each point uses only the `window` observations up to that date, so the series is what the
    estimate would have been at the time rather than a figure computed with hindsight. That
    distinction is what makes the series usable for the Phase 12 backtest: a VaR series
    contaminated by future data would pass any breach test trivially.
    """
    quantile_level = 1.0 - confidence

    def one_window(values: np.ndarray) -> float:
        return -(np.exp(np.quantile(values, quantile_level)) - 1.0)

    return returns.dropna().rolling(window=window, min_periods=window).apply(one_window, raw=True)


def breaches(returns: pd.Series, var_series: pd.Series) -> pd.DataFrame:
    """Days where the realised loss exceeded that day's VaR estimate.

    The count is the headline backtest number (Phase 12 formalises it with Kupiec and
    Christoffersen tests): a well-calibrated 95% VaR should be breached on about 5% of days,
    and materially fewer breaches is a model that is too conservative rather than a safe one.
    Both are failures of calibration, in opposite directions.
    """
    aligned = pd.DataFrame({"return": returns, "var": var_series}).dropna()
    if aligned.empty:
        return pd.DataFrame(columns=["date", "realised_loss", "var", "excess"])

    realised_loss = -(np.exp(aligned["return"]) - 1.0)
    is_breach = realised_loss > aligned["var"]

    out = pd.DataFrame({
        "date": aligned.index[is_breach],
        "realised_loss": realised_loss[is_breach].to_numpy(),
        "var": aligned["var"][is_breach].to_numpy(),
    })
    out["excess"] = out["realised_loss"] - out["var"]
    return out.reset_index(drop=True)


def breach_rate(returns: pd.Series, var_series: pd.Series) -> dict:
    """Observed breach rate against the rate the confidence level implies."""
    aligned = pd.DataFrame({"return": returns, "var": var_series}).dropna()
    n = len(aligned)
    if n == 0:
        return {"n_days": 0, "n_breaches": 0, "observed_rate": float("nan")}
    n_breaches = len(breaches(returns, var_series))
    return {
        "n_days": n,
        "n_breaches": n_breaches,
        "observed_rate": n_breaches / n,
    }
