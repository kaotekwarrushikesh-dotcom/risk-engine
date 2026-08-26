"""Phase 11: portfolio risk.

Everything before this phase measures one security at a time. A portfolio is not the sum of
its parts, and the gap between those two things is the entire subject:

    sigma_p = sqrt(w' * Sigma * w)

Portfolio variance is a quadratic form, not a weighted average. Two assets each with 25%
volatility combine into a portfolio with less than 25% volatility unless they are perfectly
correlated, and that reduction is the only thing in finance that is genuinely free. This
module measures how much of it a given portfolio is actually getting.

**Simple returns, not log returns.** The rest of the engine works in log returns because they
add across *time*, which is what volatility scaling and GARCH need. A portfolio return is the
weighted sum of its holdings' returns on the same day, which is addition across *assets*, and
log returns are not additive that way. Using them here would be wrong in a manner that is
small enough to pass casual inspection and grows with volatility, so this module converts and
a test pins the difference.

**Risk contribution is where intuition fails most often.** A holding's share of portfolio risk
is not its weight. It is its weight times its *marginal* contribution, which depends on how it
correlates with everything else. A 5% position in something that moves against the rest can
carry negative risk contribution: it is reducing total risk, not adding to it. The
contributions sum exactly to portfolio volatility (Euler's theorem for homogeneous functions),
which gives a hard arithmetic check that the decomposition is right rather than merely
plausible.

**Concentration is measured, not eyeballed.** Ten holdings where one is 80% of the book is not
a diversified portfolio, and a holdings count says nothing about that. The Herfindahl index
and the effective number of positions it implies say it in one figure.

**Sharpe and Sortino disagree on purpose.** Sharpe divides excess return by total volatility,
penalising upside moves exactly as much as downside ones, which is not how anyone actually
experiences risk. Sortino divides by downside deviation only. Where they diverge, the return
distribution is asymmetric, and that gap is information rather than noise.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
# Below this many overlapping observations a covariance matrix is too unstable to trust: with
# fewer days than assets it is not even invertible, and near that boundary it is noise.
MIN_OVERLAP = 60
# A correlation this high between two holdings means they are close to the same bet.
HIGH_CORRELATION = 0.90
# Effective N below this share of the holdings count means concentration is doing real work.
CONCENTRATION_WARNING = 0.5


@dataclass(frozen=True)
class PortfolioRisk:
    """One portfolio's risk decomposition."""

    tickers: tuple[str, ...]
    weights: np.ndarray
    n_observations: int

    annualised_volatility: float
    annualised_return: float
    weighted_average_volatility: float

    correlation: pd.DataFrame
    covariance: pd.DataFrame

    marginal_contribution: pd.Series
    risk_contribution: pd.Series
    percent_contribution: pd.Series

    herfindahl: float
    effective_holdings: float

    sharpe: float
    sortino: float
    max_drawdown: float

    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def diversification_ratio(self) -> float:
        """Weighted average volatility divided by portfolio volatility.

        1.0 means no diversification benefit at all, which happens only when every holding is
        perfectly correlated. Higher is better, and the excess over 1.0 is precisely the risk
        that cancelled out rather than being paid for.
        """
        if self.annualised_volatility <= 0:
            return float("nan")
        return self.weighted_average_volatility / self.annualised_volatility

    @property
    def risk_reduction(self) -> float:
        """The share of weighted-average volatility that diversification removed."""
        if self.weighted_average_volatility <= 0:
            return float("nan")
        return 1.0 - self.annualised_volatility / self.weighted_average_volatility

    def summary(self) -> str:
        return (
            f"{len(self.tickers)} holdings, {self.effective_holdings:.1f} effective. "
            f"Volatility {self.annualised_volatility:.1%} against a weighted average of "
            f"{self.weighted_average_volatility:.1%}, so diversification removed "
            f"{self.risk_reduction:.1%}. Sharpe {self.sharpe:.2f}, Sortino {self.sortino:.2f}."
        )


def normalise_weights(weights: dict[str, float] | np.ndarray, tickers: list[str]) -> np.ndarray:
    """Weights as a vector summing to 1.

    Negative weights are allowed and mean a short position, which is a real portfolio and not
    an input error. Weights summing to zero are refused, because scaling them to sum to one is
    then undefined rather than merely awkward.
    """
    if isinstance(weights, dict):
        missing = set(tickers) - set(weights)
        if missing:
            raise ValueError(f"no weight given for {sorted(missing)}")
        vector = np.array([float(weights[t]) for t in tickers])
    else:
        vector = np.asarray(weights, dtype=float)
        if len(vector) != len(tickers):
            raise ValueError(f"{len(vector)} weights for {len(tickers)} tickers")

    total = vector.sum()
    if np.isclose(total, 0.0):
        raise ValueError("weights sum to zero, so they cannot be scaled to sum to one")
    return vector / total


def align_returns(returns_by_ticker: dict[str, pd.Series]) -> pd.DataFrame:
    """One frame of simple returns on the dates every holding has in common.

    Inner join rather than forward fill. A holding that did not trade on a date has no return
    that day, and inventing a zero would understate its volatility and its correlation with
    everything else, which flatters the portfolio in both directions at once.
    """
    if len(returns_by_ticker) < 2:
        raise ValueError("a portfolio needs at least two holdings")

    frame = pd.DataFrame(returns_by_ticker).dropna()
    if len(frame) < MIN_OVERLAP:
        raise ValueError(
            f"only {len(frame)} dates common to all holdings, below the {MIN_OVERLAP} needed "
            "for a usable covariance matrix. A holding with a short history, or one listed in "
            "a market with different trading days, is the usual cause."
        )
    return frame


def to_simple_returns(log_returns: pd.Series | pd.DataFrame):
    """exp(r) - 1. Portfolio aggregation is across assets, where log returns do not add."""
    return np.exp(log_returns) - 1.0


def analyse_portfolio(
    returns_by_ticker: dict[str, pd.Series],
    weights: dict[str, float] | np.ndarray,
    risk_free_rate: float = 0.0,
    already_simple: bool = False,
) -> PortfolioRisk:
    """Full risk decomposition for one portfolio.

    `returns_by_ticker` holds log returns by default, matching the rest of the engine, and is
    converted internally. `risk_free_rate` is annualised, used only by Sharpe and Sortino.
    """
    frame = align_returns(returns_by_ticker)
    if not already_simple:
        frame = to_simple_returns(frame)

    tickers = list(frame.columns)
    w = normalise_weights(weights, tickers)
    n = len(frame)
    warnings: list[str] = []
    notes: list[str] = []

    covariance = frame.cov() * TRADING_DAYS_PER_YEAR
    correlation = frame.corr()

    variance = float(w @ covariance.to_numpy() @ w)
    volatility = float(np.sqrt(max(variance, 0.0)))

    individual_volatility = frame.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    weighted_average_volatility = float(np.abs(w) @ individual_volatility.to_numpy())

    portfolio_returns = pd.Series(frame.to_numpy() @ w, index=frame.index)
    annualised_return = float(portfolio_returns.mean() * TRADING_DAYS_PER_YEAR)

    # Marginal contribution: the derivative of portfolio volatility with respect to each
    # weight. Risk contribution is that times the weight, and by Euler's theorem the
    # contributions sum exactly to portfolio volatility, which is asserted below.
    if volatility > 0:
        marginal = (covariance.to_numpy() @ w) / volatility
    else:
        marginal = np.zeros_like(w)
    contribution = w * marginal
    percent = contribution / volatility if volatility > 0 else np.zeros_like(w)

    if volatility > 0:
        assert np.isclose(contribution.sum(), volatility, rtol=1e-8), (
            "risk contributions must sum to portfolio volatility; a mismatch means the "
            "decomposition is wrong rather than merely imprecise"
        )

    herfindahl = float(np.sum(w**2))
    effective = 1.0 / herfindahl if herfindahl > 0 else float("nan")

    excess = portfolio_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    daily_sigma = float(portfolio_returns.std(ddof=1))
    sharpe = (float(excess.mean()) / daily_sigma * np.sqrt(TRADING_DAYS_PER_YEAR)
              if daily_sigma > 0 else float("nan"))

    # Sortino uses downside deviation, measured against the target rather than against the
    # mean, and divides by the full observation count rather than only the losing days. That
    # second choice matters: dividing by the count of losing days alone would make a portfolio
    # look better simply for having fewer of them.
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)))
    sortino = (float(excess.mean()) / downside_deviation * np.sqrt(TRADING_DAYS_PER_YEAR)
               if downside_deviation > 0 else float("nan"))

    curve = (1.0 + portfolio_returns).cumprod()
    max_drawdown = float((curve / curve.cummax() - 1.0).min())

    # --- diagnostics the numbers alone do not surface ---------------------------------------

    if effective < len(tickers) * CONCENTRATION_WARNING:
        heaviest = tickers[int(np.argmax(np.abs(w)))]
        warnings.append(
            f"{len(tickers)} holdings but an effective count of only {effective:.1f}. "
            f"Concentration is doing real work here, with {heaviest} at "
            f"{w[int(np.argmax(np.abs(w)))]:.1%}. A holdings count is not diversification."
        )

    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    pairs = upper.stack()
    for (a, b), value in pairs[pairs > HIGH_CORRELATION].items():
        warnings.append(
            f"{a} and {b} correlate at {value:.2f}. At this level they are close to the same "
            "bet, and holding both provides much less diversification than two names suggest."
        )

    negative = [t for t, c in zip(tickers, contribution) if c < 0]
    if negative:
        notes.append(
            f"{', '.join(negative)} contributes negative risk: it moves against the rest of "
            "the book, so it reduces total portfolio volatility rather than adding to it. "
            "This is why risk contribution is not the same thing as weight."
        )

    if any(x < 0 for x in w):
        notes.append(
            "This portfolio contains short positions. Weighted-average volatility uses "
            "absolute weights, since a short position contributes risk regardless of sign."
        )

    if not np.isnan(sharpe) and not np.isnan(sortino) and sortino > sharpe * 1.3:
        notes.append(
            f"Sortino ({sortino:.2f}) sits well above Sharpe ({sharpe:.2f}), which means the "
            "volatility being penalised is disproportionately upside. Sharpe treats a good "
            "day and a bad day of the same size identically; this gap is where that "
            "assumption is costing the portfolio credit it has earned."
        )

    notes.append(
        f"Computed on {n} dates common to all {len(tickers)} holdings. Risk contributions sum "
        f"to portfolio volatility by construction and are asserted to do so at runtime."
    )

    return PortfolioRisk(
        tickers=tuple(tickers), weights=w, n_observations=n,
        annualised_volatility=volatility, annualised_return=annualised_return,
        weighted_average_volatility=weighted_average_volatility,
        correlation=correlation, covariance=covariance,
        marginal_contribution=pd.Series(marginal, index=tickers),
        risk_contribution=pd.Series(contribution, index=tickers),
        percent_contribution=pd.Series(percent, index=tickers),
        herfindahl=herfindahl, effective_holdings=effective,
        sharpe=sharpe, sortino=sortino, max_drawdown=max_drawdown,
        warnings=tuple(warnings), notes=tuple(notes),
    )


def portfolio_returns_series(
    returns_by_ticker: dict[str, pd.Series],
    weights: dict[str, float] | np.ndarray,
    already_simple: bool = False,
) -> pd.Series:
    """The portfolio's own return series, for feeding the single-asset VaR and ES tools.

    Returned as log returns so it drops straight into the rest of the engine, which expects
    them. The aggregation happens in simple space, where it is correct, and converts back.
    """
    frame = align_returns(returns_by_ticker)
    if not already_simple:
        frame = to_simple_returns(frame)
    w = normalise_weights(weights, list(frame.columns))
    simple = pd.Series(frame.to_numpy() @ w, index=frame.index)
    # A portfolio can in principle lose more than 100% with leverage or shorts, where the log
    # is undefined. Those observations are dropped rather than clipped, since clipping would
    # invent a survivable day that did not happen.
    return np.log(1.0 + simple[simple > -1.0])


def equal_weights(tickers: list[str]) -> dict[str, float]:
    """The honest default. Equal weighting beats most weighting schemes out of sample, and
    it makes no claim to know which holding deserves more."""
    return {t: 1.0 / len(tickers) for t in tickers}


def minimum_variance_weights(returns_by_ticker: dict[str, pd.Series],
                             already_simple: bool = False) -> dict[str, float]:
    """Weights minimising portfolio variance, the one point on the efficient frontier that
    needs no return forecast.

    Every other point requires expected returns, which are estimated with far more error than
    covariances and dominate the result. Minimum variance is offered here precisely because it
    avoids that: it is the frontier portfolio you can compute without pretending to know which
    asset will do best. Short positions are allowed, since constraining them is a separate
    decision the caller should make explicitly.
    """
    frame = align_returns(returns_by_ticker)
    if not already_simple:
        frame = to_simple_returns(frame)

    covariance = frame.cov().to_numpy() * TRADING_DAYS_PER_YEAR
    ones = np.ones(len(frame.columns))
    try:
        inverse = np.linalg.pinv(covariance)
    except np.linalg.LinAlgError as exc:  # noqa: BLE001
        raise ValueError(f"covariance matrix could not be inverted: {exc}") from exc

    raw = inverse @ ones
    denominator = ones @ raw
    if np.isclose(denominator, 0.0):
        raise ValueError("degenerate covariance matrix: minimum-variance weights are undefined")
    return dict(zip(frame.columns, raw / denominator))
