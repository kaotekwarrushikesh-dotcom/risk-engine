"""Tests for Phase 12: VaR backtesting.

The two tests measure genuinely different properties, and the tests here are built around
that: a model with the right breach *count* can still be badly wrong if the breaches arrive
together, and a construction where that happens is the clearest way to prove the
independence test does what it claims.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk.backtesting import (
    LOW_POWER_OBSERVATIONS,
    backtest_var,
    basel_traffic_light,
    christoffersen_independence_test,
    compare_backtests,
    kupiec_test,
    max_consecutive,
)


def returns_and_var(breach_days: list[int], n: int = 1000, var_level: float = 0.03):
    """A return series that breaches a flat VaR on exactly the days named."""
    idx = pd.bdate_range("2019-01-01", periods=n)
    values = np.full(n, 0.001)
    for day in breach_days:
        values[day] = np.log(1 - (var_level + 0.02))  # a loss comfortably past the threshold
    returns = pd.Series(values, index=idx)
    var_series = pd.Series(var_level, index=idx)
    return returns, var_series


# --- Kupiec: is the breach count right? -------------------------------------------------------

def test_a_correctly_calibrated_count_is_not_rejected():
    statistic, p = kupiec_test(n=1000, breaches=50, expected_rate=0.05)
    assert statistic == pytest.approx(0.0, abs=1e-9)
    assert p > 0.99


def test_far_too_many_breaches_are_rejected():
    _, p = kupiec_test(n=1000, breaches=150, expected_rate=0.05)
    assert p < 0.01


def test_far_too_few_breaches_are_also_rejected():
    """Over-conservatism is a calibration failure too: capital held against a loss that never
    comes is capital doing nothing."""
    _, p = kupiec_test(n=1000, breaches=5, expected_rate=0.05)
    assert p < 0.01


def test_the_zero_breach_boundary_is_handled_without_taking_log_of_zero():
    statistic, p = kupiec_test(n=500, breaches=0, expected_rate=0.05)
    assert np.isfinite(statistic)
    assert statistic == pytest.approx(-2.0 * 500 * np.log(0.95))


def test_the_all_breach_boundary_is_handled_too():
    statistic, p = kupiec_test(n=100, breaches=100, expected_rate=0.05)
    assert np.isfinite(statistic)


def test_an_empty_sample_returns_nan_rather_than_dividing_by_zero():
    statistic, p = kupiec_test(n=0, breaches=0, expected_rate=0.05)
    assert np.isnan(statistic) and np.isnan(p)


# --- Christoffersen: are the breaches spread out? ------------------------------------------------

def test_clustered_breaches_are_rejected_for_dependence():
    """The failure mode an unconditional VaR is expected to show: a fixed threshold cannot
    rise during a stress period, so its breaches bunch into the weeks it failed to anticipate."""
    clustered = pd.Series([0] * 200 + [1] * 20 + [0] * 780)
    _, p = christoffersen_independence_test(clustered)
    assert p < 0.01


def test_randomly_placed_breaches_are_not_rejected():
    """Independence means randomly placed, not evenly spaced. A perfectly periodic pattern is
    dependent in the other direction (a breach is never followed by one), and the test
    correctly rejects that too, which is why the fixture here is random rather than regular."""
    rng = np.random.default_rng(4)
    independent = pd.Series((rng.random(2000) < 0.05).astype(int))
    _, p = christoffersen_independence_test(independent)
    assert p > 0.05


def test_perfectly_periodic_breaches_are_rejected_as_dependent_too():
    """Anti-clustering is dependence: after a breach in a period-20 pattern, tomorrow is
    guaranteed not to be one, and the Markov test detects lag-1 structure in either direction."""
    periodic = pd.Series([1 if i % 20 == 0 else 0 for i in range(1000)])
    _, p = christoffersen_independence_test(periodic)
    assert p < 0.05


def test_the_independence_test_ignores_the_breach_count_entirely():
    """Same number of breaches, opposite verdicts. This is what makes it an independent
    property rather than a restatement of Kupiec."""
    n_breaches = 20
    clustered = pd.Series([0] * 200 + [1] * n_breaches + [0] * (1000 - 200 - n_breaches))
    spread = pd.Series([1 if i % 50 == 0 else 0 for i in range(1000)])

    assert clustered.sum() == spread.sum() == n_breaches
    _, clustered_p = christoffersen_independence_test(clustered)
    _, spread_p = christoffersen_independence_test(spread)
    assert clustered_p < 0.01
    assert spread_p > 0.05


def test_no_breaches_at_all_does_not_reject_for_clustering():
    """With no breaches the alternative model is unidentified; there is no clustering to
    detect, so not rejecting is correct rather than an error."""
    statistic, p = christoffersen_independence_test(pd.Series([0] * 500))
    assert statistic == 0.0 and p == 1.0


def test_a_single_breach_does_not_reject_for_clustering():
    statistic, p = christoffersen_independence_test(pd.Series([0] * 499 + [1]))
    assert np.isfinite(statistic) and p >= 0.05


def test_the_statistic_is_never_negative():
    for series in (pd.Series([0, 1] * 250), pd.Series([1] * 10 + [0] * 490)):
        statistic, _ = christoffersen_independence_test(series)
        assert statistic >= 0.0


# --- Run length ------------------------------------------------------------------------------------

def test_max_consecutive_counts_the_longest_run():
    assert max_consecutive(pd.Series([0, 1, 1, 1, 0, 1, 1, 0])) == 3
    assert max_consecutive(pd.Series([0, 0, 0])) == 0
    assert max_consecutive(pd.Series([1, 1])) == 2


# --- The full backtest ------------------------------------------------------------------------------

def test_a_model_can_pass_on_count_and_fail_on_clustering():
    """The central point of running both tests, demonstrated end to end."""
    breach_days = list(range(300, 310))  # ten breaches, all consecutive
    returns, var_series = returns_and_var(breach_days, n=1000)
    result = backtest_var(returns, var_series, confidence=0.99)

    assert result.n_breaches == 10
    assert result.passes_coverage       # 10 in 1000 is exactly the expected 1%
    assert not result.passes_independence
    assert "arriving together" in result.verdict


def test_a_well_behaved_model_passes_both():
    breach_days = list(range(0, 1000, 100))  # ten breaches, evenly spread
    returns, var_series = returns_and_var(breach_days, n=1000)
    result = backtest_var(returns, var_series, confidence=0.99)
    assert result.passes_coverage and result.passes_independence
    assert result.verdict == "not rejected on either property"


def test_the_joint_statistic_is_the_sum_of_the_two():
    returns, var_series = returns_and_var(list(range(0, 1000, 60)), n=1000)
    r = backtest_var(returns, var_series, confidence=0.99)
    assert r.conditional_coverage_statistic == pytest.approx(
        r.kupiec_statistic + r.christoffersen_statistic)


def test_a_small_sample_is_flagged_as_low_power():
    """'Not rejected' at 200 observations is a much weaker statement than it looks."""
    returns, var_series = returns_and_var([50, 120], n=200)
    result = backtest_var(returns, var_series, confidence=0.99)
    assert result.n_observations < LOW_POWER_OBSERVATIONS
    assert any("low power" in w for w in result.warnings)


def test_zero_breaches_is_reported_as_a_failure_not_a_success():
    returns, var_series = returns_and_var([], n=1000)
    result = backtest_var(returns, var_series, confidence=0.99)
    assert result.n_breaches == 0
    assert any("far too conservative" in w for w in result.warnings)


def test_no_overlap_is_refused_rather_than_reported_as_a_pass():
    returns = pd.Series([0.01], index=pd.bdate_range("2020-01-01", periods=1))
    var_series = pd.Series([0.03], index=pd.bdate_range("2021-01-01", periods=1))
    with pytest.raises(ValueError, match="no overlapping days"):
        backtest_var(returns, var_series)


# --- Basel traffic light ---------------------------------------------------------------------------

def test_the_basel_zones_match_the_published_thresholds():
    assert basel_traffic_light(4)["zone"] == "green"
    assert basel_traffic_light(5)["zone"] == "yellow"
    assert basel_traffic_light(9)["zone"] == "yellow"
    assert basel_traffic_light(10)["zone"] == "red"


def test_a_non_standard_window_is_scaled_and_says_so():
    exact = basel_traffic_light(4, n_observations=250)
    scaled = basel_traffic_light(8, n_observations=500)
    assert exact["exact_window"] is True
    assert scaled["exact_window"] is False
    assert scaled["green_max"] == pytest.approx(8.0)


# --- The comparison table ------------------------------------------------------------------------------

def test_comparison_puts_each_models_two_properties_side_by_side():
    returns, flat_var = returns_and_var(list(range(300, 310)), n=1000)
    _, spread_var = returns_and_var(list(range(0, 1000, 100)), n=1000)

    frame = compare_backtests(
        returns, {"clustered": flat_var, "spread": spread_var}, confidence=0.99)

    assert len(frame) == 2
    assert {"coverage", "independence", "kupiec_p", "christoffersen_p"} <= set(frame.columns)
    clustered_row = frame.loc[frame["model"] == "clustered"].iloc[0]
    assert clustered_row["coverage"] == "pass"
    assert clustered_row["independence"] == "FAIL"
