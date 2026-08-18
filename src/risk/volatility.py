"""Phase 2b: historical and rolling volatility.

Volatility here means the standard deviation of returns, annualised by the standard
square-root-of-time scaling: `sigma_annual = sigma_period * sqrt(periods_per_year)`. That
scaling assumes returns are independent and identically distributed period to period, which
real returns are not (volatility clusters, which is the entire reason Phase 8's GARCH model
exists), so this is a working approximation rather than a claim that volatility is constant.
Daily volatility annualises on 252 trading days, weekly on 52, monthly on 12.

**The rolling window is a real modelling choice, not a display setting.** A 20-day window
reacts fast to a regime change and is noisy; a 252-day window is stable and slow to reflect
one. Both are legitimate depending on what the number is for, which is why the window is a
parameter the caller sets rather than a constant buried in the function.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class VolatilitySummary:
    """Volatility at every standard horizon, from one return series."""

    daily: float
    weekly: float
    monthly: float
    annualised: float
    window: int
    n_observations: int


def annualise_volatility(period_vol: float, periods_per_year: int) -> float:
    """sigma_annual = sigma_period x sqrt(periods_per_year)."""
    return period_vol * np.sqrt(periods_per_year)


def historical_volatility(returns: pd.Series, window: int | None = None) -> VolatilitySummary:
    """Volatility at daily, weekly, monthly and annualised horizons.

    `window` restricts the calculation to the most recent N observations, so a caller asking
    "what has volatility looked like over the last 60 days" gets exactly that rather than the
    volatility over the whole available history.
    """
    sample = returns.tail(window) if window else returns
    n = len(sample)

    daily_sigma = float(sample.std(ddof=1)) if n > 1 else float("nan")

    # Weekly and monthly figures scale the daily standard deviation to that horizon (5 and
    # 21 trading days respectively); "annualised" is the same square-root-of-time scaling
    # taken all the way to a full trading year. All three are the same operation at a
    # different horizon, not three different calculations.
    weekly_sigma = daily_sigma * np.sqrt(5) if not np.isnan(daily_sigma) else float("nan")
    monthly_sigma = daily_sigma * np.sqrt(21) if not np.isnan(daily_sigma) else float("nan")

    return VolatilitySummary(
        daily=daily_sigma,
        weekly=weekly_sigma,
        monthly=monthly_sigma,
        annualised=annualise_volatility(daily_sigma, TRADING_DAYS_PER_YEAR)
        if not np.isnan(daily_sigma) else float("nan"),
        window=window or n,
        n_observations=n,
    )


def rolling_volatility(returns: pd.Series, window: int, annualise: bool = True) -> pd.Series:
    """A rolling annualised volatility series, for charting how risk has moved over time.

    The first `window - 1` points are NaN by construction: there is no such thing as a
    30-day rolling volatility on day 5, and padding it with a placeholder would misrepresent
    the reader's confidence in an early-series number that does not exist.
    """
    rolling_sigma = returns.rolling(window=window, min_periods=window).std(ddof=1)
    if annualise:
        return rolling_sigma * np.sqrt(TRADING_DAYS_PER_YEAR)
    return rolling_sigma


def volatility_regime(current: float, historical_average: float) -> str:
    """A plain-language read on whether current volatility is elevated.

    Thresholds are ratios to the series' own historical average rather than fixed absolute
    levels, since "high volatility" means something different for a utility than for a
    speculative small-cap; comparing a stock to its own history sidesteps that.
    """
    if np.isnan(current) or np.isnan(historical_average) or historical_average == 0:
        return "unknown"
    ratio = current / historical_average
    if ratio >= 1.75:
        return "elevated"
    if ratio <= 0.6:
        return "subdued"
    return "normal"


def identify_volatility_spikes(rolling_vol: pd.Series, threshold_multiple: float = 1.75) -> pd.DataFrame:
    """Periods where rolling volatility exceeded a multiple of its own series median.

    Consecutive spike days are grouped into episodes rather than listed individually, since
    a 40-day period of sustained stress reads as one event to an analyst, not forty.
    """
    valid = rolling_vol.dropna()
    if valid.empty:
        return pd.DataFrame(columns=["start", "end", "peak_volatility", "duration_days"])

    median = float(valid.median())
    is_spike = valid > median * threshold_multiple

    episodes = []
    in_spike, start = False, None
    for date, flag in is_spike.items():
        if flag and not in_spike:
            in_spike, start = True, date
        elif not flag and in_spike:
            segment = valid.loc[start:date]
            episodes.append({
                "start": start, "end": segment.index[-2] if len(segment) > 1 else start,
                "peak_volatility": float(segment.max()), "duration_days": len(segment) - 1,
            })
            in_spike = False
    if in_spike:
        segment = valid.loc[start:]
        episodes.append({
            "start": start, "end": segment.index[-1],
            "peak_volatility": float(segment.max()), "duration_days": len(segment),
        })

    return pd.DataFrame(episodes)
