"""Tests for Phase 7: Expected Shortfall and the methods comparison.

The subtle failure this file exists to prevent is the tail-selection bug: defining the tail
as `values <= quantile` looks equivalent to taking the worst k observations and is not. On a
distribution with an atom the comparison can select the entire sample, so the "5% ES"
silently becomes the mean of everything, which is not obviously wrong on inspection and is
not a tail statistic at all.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_engine.expected_shortfall import (
    compare_methods,
    demonstrate_var_subadditivity_failure,
    empirical_tail,
    historical_expected_shortfall,
    monte_carlo_expected_shortfall,
    parametric_expected_shortfall,
    tail_observations_for,
)


def normal_returns(n=2000, seed=7, mu=0.0004, sigma=0.011) -> pd.Series:
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(np.random.default_rng(seed).normal(mu, sigma, n), index=idx)


def fat_tailed_returns(n=2000, seed=7, df=3.0, sigma=0.011) -> pd.Series:
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(seed)
    raw = rng.standard_t(df, n)
    return pd.Series(raw / np.sqrt(df / (df - 2)) * sigma, index=idx)


# --- Tail selection, which is where the quiet bug lives -------------------------------------

def test_tail_size_is_the_ceiling_of_n_times_one_minus_confidence():
    assert tail_observations_for(1000, 0.95) == 50
    assert tail_observations_for(1000, 0.99) == 10
    assert tail_observations_for(251, 0.99) == 3  # ceil(2.51)


def test_a_tail_is_never_empty():
    assert tail_observations_for(10, 0.99) == 1


def test_the_tail_is_the_worst_k_observations():
    values = np.array([5.0, -1.0, 3.0, -4.0, 0.0, 2.0, -2.0, 1.0, -3.0, 4.0])
    tail = empirical_tail(values, 0.80)  # worst 2 of 10
    assert sorted(tail) == [-4.0, -3.0]


def test_an_atom_at_the_quantile_does_not_swallow_the_whole_sample():
    """The bug this guards: 4% of mass at a large loss and 96% at a small gain puts the 5%
    quantile *on* the gain, so `values <= quantile` selects everything and the ES becomes the
    mean of the distribution rather than of its tail."""
    values = np.where(np.arange(10_000) < 400, -1.0, 0.02)

    naive_tail = values[values <= np.quantile(values, 0.05)]
    assert len(naive_tail) == len(values)  # the bug, demonstrated

    tail = empirical_tail(values, 0.95)
    assert len(tail) == 500
    assert float(np.mean(tail)) == pytest.approx((400 * -1.0 + 100 * 0.02) / 500)


# --- Historical ES ------------------------------------------------------------------------------

def test_es_is_always_worse_than_the_var_it_sits_behind():
    """ES averages losses *beyond* the threshold, so it cannot be smaller than it."""
    for confidence in (0.90, 0.95, 0.99):
        r = historical_expected_shortfall(fat_tailed_returns(), confidence)
        assert r.es_return > r.var_return
        assert r.tail_severity > 1.0


def test_historical_es_matches_a_hand_computed_tail_average():
    returns = pd.Series(np.linspace(-0.10, 0.10, 200))
    r = historical_expected_shortfall(returns, confidence=0.95)
    worst_ten = np.sort(returns.to_numpy())[:10]
    assert r.es_return == pytest.approx(-(np.exp(worst_ten.mean()) - 1.0))


def test_es_rests_on_fewer_observations_than_var_and_says_so():
    r = historical_expected_shortfall(normal_returns(n=200), confidence=0.99)
    assert r.tail_observations <= 2
    assert any("average of a handful" in w for w in r.warnings)


def test_fat_tails_widen_the_gap_between_es_and_var():
    """The whole reason ES exists: two series can share a VaR and differ past it."""
    thin = historical_expected_shortfall(normal_returns(), 0.99).tail_severity
    fat = historical_expected_shortfall(fat_tailed_returns(), 0.99).tail_severity
    assert fat > thin


# --- Parametric ES against its own closed form ------------------------------------------------

def test_normal_es_closed_form_matches_a_large_simulation():
    """The analytic result is used instead of averaging draws; this checks they agree."""
    mu, sigma = 0.0005, 0.012
    returns = pd.Series(stats.norm.rvs(mu, sigma, size=200_000, random_state=3))
    r = parametric_expected_shortfall(returns, 0.95, distribution="normal")

    draws = returns.to_numpy()
    simulated_es = -(np.exp(np.mean(empirical_tail(draws, 0.95))) - 1.0)
    assert r.es_return == pytest.approx(simulated_es, rel=0.02)


def test_student_t_es_is_rescaled_to_unit_variance():
    """Omitting sqrt((v-2)/v) inflates ES exactly as it inflates VaR."""
    returns = fat_tailed_returns()
    r = parametric_expected_shortfall(returns, 0.99, distribution="t")
    assert r.es_return > r.var_return
    # Sanity: a t-ES on fat-tailed data must exceed the normal ES on the same data.
    normal = parametric_expected_shortfall(returns, 0.99, distribution="normal")
    assert r.es_return > normal.es_return


def test_normal_es_understates_the_tail_of_fat_tailed_data():
    returns = fat_tailed_returns()
    normal = parametric_expected_shortfall(returns, 0.99, distribution="normal").es_return
    empirical = historical_expected_shortfall(returns, 0.99).es_return
    assert normal < empirical


def test_an_unknown_distribution_is_refused():
    with pytest.raises(ValueError, match="unknown distribution"):
        parametric_expected_shortfall(normal_returns(), distribution="cauchy")


# --- Monte Carlo ES ------------------------------------------------------------------------------

def test_bootstrap_es_reproduces_the_historical_es():
    returns = fat_tailed_returns()
    empirical = historical_expected_shortfall(returns, 0.95).es_return
    simulated = monte_carlo_expected_shortfall(
        returns, 0.95, draw_method="bootstrap", paths=50_000).es_return
    assert simulated == pytest.approx(empirical, rel=0.10)


def test_monte_carlo_es_exceeds_its_own_var():
    r = monte_carlo_expected_shortfall(fat_tailed_returns(), 0.99, paths=20_000)
    assert r.es_return > r.var_return


# --- The coherence argument, demonstrated rather than asserted -------------------------------

def test_var_can_fail_subadditivity_while_es_does_not():
    """Diversification cannot increase risk. VaR says it did; that is a defect of the
    measure, and it is why Basel moved to ES for market-risk capital."""
    d = demonstrate_var_subadditivity_failure()
    assert d["var_is_subadditive"] is False
    assert d["var_combined"] > d["var_sum_of_parts"]
    assert d["es_is_subadditive"] is True
    assert d["es_combined"] <= d["es_sum_of_parts"]


# --- The comparison table --------------------------------------------------------------------

def test_comparison_covers_every_method_built():
    frame = compare_methods(normal_returns(), 0.95, paths=5_000)
    assert len(frame) == 5
    assert {"method", "assumption", "var", "es", "tail_severity"} <= set(frame.columns)
    assert frame["es"].gt(frame["var"]).all()


def test_comparison_is_measured_against_the_historical_baseline():
    frame = compare_methods(normal_returns(), 0.95, paths=5_000)
    assert frame["method"].iloc[0] == "Historical"
    assert frame["var_vs_historical"].iloc[0] == pytest.approx(0.0)


def test_the_normal_row_sits_below_the_historical_row_on_fat_tailed_data_at_99():
    frame = compare_methods(fat_tailed_returns(), 0.99, paths=20_000)
    historical = frame.loc[frame["method"] == "Historical", "es"].iloc[0]
    normal = frame.loc[frame["method"] == "Parametric (normal)", "es"].iloc[0]
    assert normal < historical
