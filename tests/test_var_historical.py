"""Tests for Phase 4: historical VaR.

The properties worth pinning are the ones that would make a VaR number quietly wrong rather
than obviously broken: a sign convention that flips, a log return reported as if it were a
percentage loss, a tail estimate that hides how few observations it rests on, and a rolling
series contaminated by data from after the date it claims to describe.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_engine.var_historical import (
    AT_THE_EDGE_RATIO,
    MIN_TAIL_OBSERVATIONS,
    breach_rate,
    breaches,
    historical_var,
    rolling_historical_var,
    scale_to_horizon,
)


def make_returns(n=1000, seed=3, mu=0.0003, sigma=0.012) -> pd.Series:
    idx = pd.bdate_range("2021-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n), index=idx)


# --- Sign convention and units -------------------------------------------------------------

def test_var_is_reported_as_a_positive_loss():
    """A risk report that mixes positive and negative loss conventions is how a sign error
    reaches a decision, so this is pinned rather than left to convention."""
    r = historical_var(make_returns(), confidence=0.95)
    assert r.var_return > 0
    assert r.var_pct == pytest.approx(r.var_return * 100.0)


def test_log_return_is_converted_to_an_actual_loss():
    """A -5% log return is a 4.88% loss. Reporting the log figure as a percentage loss
    overstates it in a direction that looks conservative and is simply wrong."""
    # Every observation identical, so the quantile is that value regardless of level.
    flat = pd.Series([-0.05] * 100)
    r = historical_var(flat, confidence=0.95)
    assert r.var_return == pytest.approx(-(np.exp(-0.05) - 1.0))
    assert r.var_return == pytest.approx(0.04877, abs=1e-4)
    assert r.var_return < 0.05


def test_a_higher_confidence_level_gives_a_larger_loss():
    returns = make_returns()
    assert historical_var(returns, 0.99).var_return > historical_var(returns, 0.95).var_return


def test_portfolio_value_scales_the_loss_linearly():
    returns = make_returns()
    r = historical_var(returns, 0.95, portfolio_value=1_000_000)
    assert r.var_value == pytest.approx(r.var_return * 1_000_000)
    assert historical_var(returns, 0.95).var_value is None


# --- The empirical quantile itself ----------------------------------------------------------

def test_var_matches_the_empirical_quantile_by_hand():
    """No distributional assumption: the answer is a sorted-data lookup and nothing else."""
    returns = pd.Series(np.linspace(-0.10, 0.10, 101))
    r = historical_var(returns, confidence=0.95)
    expected_log_quantile = float(np.quantile(returns.to_numpy(), 0.05))
    assert r.var_return == pytest.approx(-(np.exp(expected_log_quantile) - 1.0))


def test_var_never_exceeds_the_worst_observed_loss():
    """The defining limitation of the method, asserted rather than described."""
    for confidence in (0.90, 0.95, 0.99, 0.999):
        r = historical_var(make_returns(n=500), confidence=confidence)
        assert r.var_return <= r.worst_observed_loss + 1e-12


def test_a_calm_sample_produces_a_small_var_and_says_why():
    """Historical VaR has no mechanism for a loss worse than one already seen, so on a quiet
    sample it reports a small number *because* nothing bad has happened yet."""
    calm = pd.Series(np.full(400, -0.001))
    r = historical_var(calm, confidence=0.99)
    assert r.var_return == pytest.approx(-(np.exp(-0.001) - 1.0))
    assert r.at_the_edge_of_the_data
    assert any("worse than what has already happened" in w for w in r.warnings)


def test_the_edge_flag_is_not_raised_on_a_sample_with_real_tail_room():
    r = historical_var(make_returns(n=2000), confidence=0.95)
    assert r.var_return < r.worst_observed_loss * AT_THE_EDGE_RATIO
    assert not r.at_the_edge_of_the_data


# --- Honesty about how few observations support the estimate --------------------------------

def test_tail_observation_count_is_reported():
    """A 99% VaR on 250 days rests on about 2.5 observations, and the point estimate looks
    exactly as precise as any other number unless the count travels with it."""
    r = historical_var(make_returns(n=250), confidence=0.99)
    assert r.n_observations == 250
    assert r.tail_observations <= 5
    assert any("rests on" in w for w in r.warnings)


def test_a_thin_tail_is_warned_about_and_a_thick_one_is_not():
    thin = historical_var(make_returns(n=200), confidence=0.99)
    thick = historical_var(make_returns(n=3000), confidence=0.95)
    assert thin.tail_observations < MIN_TAIL_OBSERVATIONS
    assert thick.tail_observations >= MIN_TAIL_OBSERVATIONS
    assert not any("rests on" in w for w in thick.warnings)


def test_the_confidence_interval_brackets_the_estimate_and_narrows_with_data():
    """An interval spanning 2.4% to 4.6% around a 3.1% estimate is a different statement
    from a narrow one, and the reader should not have to infer which they have."""
    small = historical_var(make_returns(n=250, seed=5), confidence=0.95)
    large = historical_var(make_returns(n=4000, seed=5), confidence=0.95)
    assert small.ci_low <= small.var_return <= small.ci_high
    assert large.ci_low <= large.var_return <= large.ci_high
    assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)


def test_results_are_reproducible_for_a_given_seed():
    a = historical_var(make_returns(), confidence=0.95, seed=42)
    b = historical_var(make_returns(), confidence=0.95, seed=42)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


# --- Horizon scaling -------------------------------------------------------------------------

def test_horizon_scaling_follows_square_root_of_time():
    assert scale_to_horizon(-0.02, 4) == pytest.approx(-0.04)
    assert scale_to_horizon(-0.02, 1) == pytest.approx(-0.02)


def test_a_ten_day_var_is_larger_than_a_one_day_var_but_not_ten_times():
    returns = make_returns()
    one = historical_var(returns, 0.95, horizon_days=1)
    ten = historical_var(returns, 0.95, horizon_days=10)
    assert ten.var_return > one.var_return
    assert ten.var_return < one.var_return * 10
    assert any("independent day to day" in n for n in ten.notes)


def test_multi_day_conversion_is_not_a_naive_multiplication_of_the_loss():
    """Scaling must happen in log space and convert once at the end; scaling the simple loss
    directly would compound the conversion error with the horizon."""
    returns = make_returns()
    one = historical_var(returns, 0.95, horizon_days=1)
    ten = historical_var(returns, 0.95, horizon_days=10)
    naive = one.var_return * np.sqrt(10)
    assert ten.var_return != pytest.approx(naive)
    assert ten.var_return < naive  # compounding a loss is sub-linear


# --- Input validation --------------------------------------------------------------------------

def test_an_impossible_confidence_level_is_refused():
    for bad in (0.4, 1.0, 1.5):
        with pytest.raises(ValueError, match="confidence"):
            historical_var(make_returns(), confidence=bad)


def test_a_horizon_below_one_day_is_refused():
    with pytest.raises(ValueError, match="horizon_days"):
        historical_var(make_returns(), horizon_days=0)


def test_too_little_data_is_refused_rather_than_estimated():
    with pytest.raises(ValueError, match="at least two"):
        historical_var(pd.Series([0.01]))


def test_missing_observations_are_dropped_not_counted():
    returns = make_returns(n=100)
    with_gaps = returns.copy()
    with_gaps.iloc[10:20] = np.nan
    assert historical_var(with_gaps).n_observations == 90


# --- Rolling VaR and breaches -------------------------------------------------------------------

def test_rolling_var_uses_only_data_available_at_the_time():
    """A VaR series contaminated by future data would pass any backtest trivially, which is
    exactly the failure a backtest exists to catch."""
    returns = make_returns(n=400)
    rolling = rolling_historical_var(returns, window=100, confidence=0.95)

    assert rolling.iloc[:99].isna().all()
    assert rolling.iloc[99:].notna().all()

    # The value on day 100 must equal a VaR computed from the first 100 observations alone.
    expected = -(np.exp(np.quantile(returns.iloc[:100].to_numpy(), 0.05)) - 1.0)
    assert rolling.iloc[99] == pytest.approx(expected)


def test_a_late_crash_does_not_change_an_early_rolling_value():
    returns = make_returns(n=300)
    crashed = returns.copy()
    crashed.iloc[-1] = -0.35
    a = rolling_historical_var(returns, window=100)
    b = rolling_historical_var(crashed, window=100)
    assert a.iloc[150] == pytest.approx(b.iloc[150])


def test_breaches_are_days_the_realised_loss_exceeded_the_estimate():
    returns = pd.Series([-0.01, -0.02, -0.09, 0.01, -0.005])
    var_series = pd.Series([0.03, 0.03, 0.03, 0.03, 0.03])
    found = breaches(returns, var_series)
    assert len(found) == 1
    assert found["realised_loss"].iloc[0] == pytest.approx(-(np.exp(-0.09) - 1.0))
    assert found["excess"].iloc[0] > 0


def test_breach_rate_is_close_to_the_level_it_claims_on_well_behaved_data():
    """A 95% VaR breached far more than 5% of the time is under-stating risk; far less is
    over-stating it. Both are calibration failures, in opposite directions."""
    returns = make_returns(n=3000, seed=11)
    rolling = rolling_historical_var(returns, window=250, confidence=0.95)
    stats = breach_rate(returns, rolling)
    assert stats["n_days"] > 2000
    assert 0.02 < stats["observed_rate"] < 0.09


def test_breach_rate_on_no_overlap_reports_nothing_rather_than_dividing_by_zero():
    stats = breach_rate(pd.Series(dtype=float), pd.Series(dtype=float))
    assert stats["n_days"] == 0
    assert np.isnan(stats["observed_rate"])
