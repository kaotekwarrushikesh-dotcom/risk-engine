"""Tests for Phase 11: portfolio risk.

The properties worth pinning are the ones where portfolio maths diverges from intuition:
that variance is a quadratic form rather than a weighted average, that risk contributions sum
exactly to portfolio volatility, that a holding can carry negative risk contribution, and
that aggregating across assets must happen in simple-return space.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_engine.portfolio import (
    HIGH_CORRELATION,
    MIN_OVERLAP,
    align_returns,
    analyse_portfolio,
    equal_weights,
    minimum_variance_weights,
    normalise_weights,
    portfolio_returns_series,
    to_simple_returns,
)

IDX = pd.bdate_range("2020-01-01", periods=800)


def correlated_returns(rho: float, n: int = 800, sigma: float = 0.015, seed: int = 5):
    """Two series with a known correlation, so the maths has a right answer to hit."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0004, sigma, n)
    noise = rng.normal(0.0004, sigma, n)
    b = rho * a + np.sqrt(max(1 - rho**2, 0.0)) * noise
    idx = pd.bdate_range("2020-01-01", periods=n)
    return {"A": pd.Series(a, index=idx), "B": pd.Series(b, index=idx)}


# --- Weights -------------------------------------------------------------------------------

def test_weights_are_scaled_to_sum_to_one():
    w = normalise_weights({"A": 2.0, "B": 2.0}, ["A", "B"])
    assert w == pytest.approx([0.5, 0.5])


def test_short_positions_are_allowed():
    """A negative weight is a real portfolio, not an input error."""
    w = normalise_weights({"A": 1.5, "B": -0.5}, ["A", "B"])
    assert w.sum() == pytest.approx(1.0)
    assert w[1] < 0


def test_weights_summing_to_zero_are_refused():
    with pytest.raises(ValueError, match="sum to zero"):
        normalise_weights({"A": 1.0, "B": -1.0}, ["A", "B"])


def test_a_missing_weight_is_named():
    with pytest.raises(ValueError, match="no weight given"):
        normalise_weights({"A": 1.0}, ["A", "B"])


def test_equal_weights_are_equal():
    assert equal_weights(["A", "B", "C", "D"]) == {t: 0.25 for t in "ABCD"}


# --- Alignment ------------------------------------------------------------------------------

def test_only_dates_common_to_every_holding_are_used():
    """Forward filling would invent a zero return, understating volatility and correlation at
    the same time, which flatters the portfolio in both directions."""
    a = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300), index=IDX[:300])
    b = pd.Series(np.random.default_rng(2).normal(0, 0.01, 300), index=IDX[100:400])
    frame = align_returns({"A": a, "B": b})
    assert len(frame) == 200


def test_a_portfolio_needs_at_least_two_holdings():
    with pytest.raises(ValueError, match="at least two holdings"):
        align_returns({"A": pd.Series([0.01] * 100)})


def test_too_little_overlap_is_refused_rather_than_estimated():
    a = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300), index=IDX[:300])
    b = pd.Series(np.random.default_rng(2).normal(0, 0.01, 300), index=IDX[280:580])
    with pytest.raises(ValueError, match="usable covariance matrix"):
        align_returns({"A": a, "B": b})


# --- The central result: variance is not a weighted average -----------------------------------

def test_imperfect_correlation_produces_a_diversification_benefit():
    """The only free lunch in finance, asserted rather than assumed."""
    r = analyse_portfolio(correlated_returns(0.2), equal_weights(["A", "B"]))
    assert r.annualised_volatility < r.weighted_average_volatility
    assert r.diversification_ratio > 1.0
    assert r.risk_reduction > 0


def test_perfect_correlation_produces_no_benefit_at_all():
    base = pd.Series(np.random.default_rng(3).normal(0.0004, 0.015, 400), index=IDX[:400])
    r = analyse_portfolio({"A": base, "B": base.copy()}, equal_weights(["A", "B"]))
    assert r.diversification_ratio == pytest.approx(1.0, abs=1e-6)
    assert r.risk_reduction == pytest.approx(0.0, abs=1e-6)


def test_lower_correlation_gives_more_diversification():
    high = analyse_portfolio(correlated_returns(0.9), equal_weights(["A", "B"]))
    low = analyse_portfolio(correlated_returns(0.1), equal_weights(["A", "B"]))
    assert low.diversification_ratio > high.diversification_ratio


def test_portfolio_volatility_matches_the_quadratic_form_by_hand():
    data = correlated_returns(0.4)
    r = analyse_portfolio(data, {"A": 0.7, "B": 0.3})
    w = np.array([0.7, 0.3])
    expected = float(np.sqrt(w @ r.covariance.to_numpy() @ w))
    assert r.annualised_volatility == pytest.approx(expected)


# --- Risk contribution ---------------------------------------------------------------------------

def test_risk_contributions_sum_exactly_to_portfolio_volatility():
    """Euler's theorem gives a hard arithmetic check that the decomposition is right rather
    than merely plausible."""
    r = analyse_portfolio(correlated_returns(0.35), {"A": 0.6, "B": 0.4})
    assert r.risk_contribution.sum() == pytest.approx(r.annualised_volatility)
    assert r.percent_contribution.sum() == pytest.approx(1.0)


def test_risk_contribution_is_not_the_same_as_weight():
    """The point of computing it. A holding's share of risk depends on how it correlates with
    everything else, not only on how much of it there is."""
    rng = np.random.default_rng(9)
    calm = pd.Series(rng.normal(0.0003, 0.004, 500), index=IDX[:500])
    wild = pd.Series(rng.normal(0.0003, 0.030, 500), index=IDX[:500])
    r = analyse_portfolio({"calm": calm, "wild": wild}, {"calm": 0.5, "wild": 0.5})
    assert r.percent_contribution["wild"] > 0.8
    assert r.percent_contribution["calm"] < 0.2


def test_a_negatively_correlated_holding_can_contribute_negative_risk():
    """It reduces total volatility rather than adding to it, which no weight-based view shows."""
    r = analyse_portfolio(correlated_returns(-0.9), {"A": 0.75, "B": 0.25})
    assert (r.risk_contribution < 0).any()
    assert any("negative risk" in n for n in r.notes)


# --- Concentration ---------------------------------------------------------------------------------

def test_effective_holdings_equals_the_count_when_equally_weighted():
    rng = np.random.default_rng(4)
    data = {t: pd.Series(rng.normal(0.0003, 0.012, 400), index=IDX[:400]) for t in "ABCD"}
    r = analyse_portfolio(data, equal_weights(list("ABCD")))
    assert r.effective_holdings == pytest.approx(4.0)
    assert r.herfindahl == pytest.approx(0.25)


def test_concentration_collapses_the_effective_count_and_is_warned_about():
    """Ten holdings where one is most of the book is not a diversified portfolio, and a
    holdings count cannot say so."""
    rng = np.random.default_rng(6)
    data = {t: pd.Series(rng.normal(0.0003, 0.012, 400), index=IDX[:400]) for t in "ABCD"}
    r = analyse_portfolio(data, {"A": 0.85, "B": 0.05, "C": 0.05, "D": 0.05})
    assert r.effective_holdings < 2.0
    assert any("effective count" in w for w in r.warnings)


def test_near_duplicate_holdings_are_flagged():
    r = analyse_portfolio(correlated_returns(0.97), equal_weights(["A", "B"]))
    assert any("close to the same bet" in w for w in r.warnings)


def test_uncorrelated_holdings_raise_no_duplication_warning():
    r = analyse_portfolio(correlated_returns(0.05), equal_weights(["A", "B"]))
    assert not any("close to the same bet" in w for w in r.warnings)


# --- Sharpe and Sortino ------------------------------------------------------------------------------

def test_sortino_exceeds_sharpe_when_downside_is_the_smaller_half():
    """They diverge on purpose: Sharpe penalises upside volatility identically to downside."""
    rng = np.random.default_rng(11)
    # Right-skewed: many small losses, occasional large gains.
    skewed = pd.Series(np.where(rng.random(600) < 0.9,
                                rng.normal(-0.001, 0.004, 600),
                                rng.normal(0.02, 0.010, 600)), index=IDX[:600])
    other = pd.Series(rng.normal(0.0004, 0.010, 600), index=IDX[:600])
    r = analyse_portfolio({"skew": skewed, "other": other}, {"skew": 0.9, "other": 0.1})
    assert r.sortino > r.sharpe


def test_a_higher_risk_free_rate_lowers_sharpe():
    data = correlated_returns(0.3)
    low = analyse_portfolio(data, equal_weights(["A", "B"]), risk_free_rate=0.0).sharpe
    high = analyse_portfolio(data, equal_weights(["A", "B"]), risk_free_rate=0.05).sharpe
    assert high < low


def test_max_drawdown_is_negative_or_zero():
    r = analyse_portfolio(correlated_returns(0.3), equal_weights(["A", "B"]))
    assert r.max_drawdown <= 0


# --- Simple versus log returns -------------------------------------------------------------------------

def test_aggregation_happens_in_simple_return_space():
    """Log returns add across time, not across assets. Using them here is wrong in a way that
    is small enough to pass inspection and grows with volatility."""
    log_a = pd.Series([0.10] * 200, index=IDX[:200])
    log_b = pd.Series([-0.05] * 200, index=IDX[:200])

    series = portfolio_returns_series({"A": log_a, "B": log_b}, {"A": 0.5, "B": 0.5})

    simple_correct = 0.5 * (np.exp(0.10) - 1) + 0.5 * (np.exp(-0.05) - 1)
    assert float(series.iloc[0]) == pytest.approx(np.log(1 + simple_correct))

    naive_log_average = 0.5 * 0.10 + 0.5 * -0.05
    assert float(series.iloc[0]) != pytest.approx(naive_log_average)


def test_the_returned_series_is_in_log_space_for_the_rest_of_the_engine():
    data = correlated_returns(0.3)
    series = portfolio_returns_series(data, equal_weights(["A", "B"]))
    assert isinstance(series, pd.Series)
    assert len(series) > 0
    # Log returns of a diversified equity portfolio stay well inside these bounds.
    assert series.abs().max() < 0.5


def test_to_simple_returns_inverts_the_log():
    assert float(to_simple_returns(pd.Series([0.0])).iloc[0]) == pytest.approx(0.0)
    assert float(to_simple_returns(pd.Series([np.log(1.1)])).iloc[0]) == pytest.approx(0.1)


# --- Minimum variance -----------------------------------------------------------------------------------

def test_minimum_variance_weights_beat_equal_weights_on_volatility():
    """This is the defining property, so it is asserted rather than trusted."""
    rng = np.random.default_rng(21)
    data = {
        "calm": pd.Series(rng.normal(0.0002, 0.005, 600), index=IDX[:600]),
        "mid": pd.Series(rng.normal(0.0004, 0.014, 600), index=IDX[:600]),
        "wild": pd.Series(rng.normal(0.0006, 0.032, 600), index=IDX[:600]),
    }
    mv = minimum_variance_weights(data)
    equal = equal_weights(list(data))
    assert (analyse_portfolio(data, mv).annualised_volatility
            < analyse_portfolio(data, equal).annualised_volatility)


def test_minimum_variance_tilts_toward_the_calmer_asset():
    rng = np.random.default_rng(22)
    data = {
        "calm": pd.Series(rng.normal(0.0002, 0.004, 600), index=IDX[:600]),
        "wild": pd.Series(rng.normal(0.0008, 0.030, 600), index=IDX[:600]),
    }
    mv = minimum_variance_weights(data)
    assert mv["calm"] > mv["wild"]


def test_minimum_variance_weights_sum_to_one():
    assert sum(minimum_variance_weights(correlated_returns(0.3)).values()) == pytest.approx(1.0)
