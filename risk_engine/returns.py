"""Phase 2a: return calculation.

    Simple:  R_t = P_t / P_(t-1) - 1
    Log:     r_t = ln(P_t / P_(t-1))

The two agree for small moves and diverge for large ones. **Log returns are the default**
for anything that aggregates across time (volatility, VaR horizon scaling, GARCH), because
they are time-additive: the log return over five days is exactly the sum of the five daily
log returns, so multi-day volatility can be built from daily volatility by simple scaling.
Simple returns are not additive across time, only across assets in a portfolio (a portfolio
return is the weight-weighted sum of simple asset returns; log returns do not have that
property), which is why portfolio-level calculations use simple returns instead.

Both are computed from `adj_close`, not `close`, so a return is never contaminated by a
dividend payment showing up as a fake single-day drop.
"""

import numpy as np
import pandas as pd

ReturnMethod = str  # "simple" | "log"


def simple_returns(prices: pd.Series) -> pd.Series:
    """R_t = P_t / P_(t-1) - 1. Additive across assets, not across time."""
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    """r_t = ln(P_t / P_(t-1)). Additive across time, not exactly across assets."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_returns(prices: pd.Series, method: ReturnMethod = "log") -> pd.Series:
    """Dispatch to the chosen methodology, so callers do not branch on a string themselves."""
    if method == "simple":
        return simple_returns(prices)
    if method == "log":
        return log_returns(prices)
    raise ValueError(f"unknown return method: {method!r}, expected 'simple' or 'log'")


def cumulative_return(returns: pd.Series, method: ReturnMethod = "log") -> float:
    """Total return over the full series, in the same units as `method`."""
    if method == "log":
        return float(returns.sum())
    return float((1.0 + returns).prod() - 1.0)


def annualise_return(mean_period_return: float, periods_per_year: int, method: ReturnMethod = "log") -> float:
    """Scale a per-period mean return to an annual figure.

    Log returns scale by simple multiplication, since they are additive across time. Simple
    returns compound, so they scale geometrically; using the log scaling on a simple-return
    mean would overstate the annual figure for anything but a small mean return.
    """
    if method == "log":
        return mean_period_return * periods_per_year
    return (1.0 + mean_period_return) ** periods_per_year - 1.0


def to_price_index(returns: pd.Series, method: ReturnMethod = "log", base: float = 100.0) -> pd.Series:
    """Rebuild a price index from a return series, for charting cumulative performance."""
    if method == "log":
        return base * np.exp(returns.cumsum())
    return base * (1.0 + returns).cumprod()
