"""Tests for Phase 6: Monte Carlo VaR.

The failure modes worth pinning: a simulation that silently loses its fat tails, a t draw
missing the unit-variance rescale, multi-day paths built by scaling rather than by actually
summing drawn days, and a result that hides how much of itself is simulation noise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk.var_historical import historical_var
from src.risk.var_monte_carlo import (
    MIN_SIMULATED_TAIL,
    convergence_path,
    monte_carlo_var,
    simulate_paths,
)


def normal_returns(n=2000, seed=7, mu=0.0004, sigma=0.011) -> pd.Series:
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(np.random.default_rng(seed).normal(mu, sigma, n), index=idx)


def fat_tailed_returns(n=2000, seed=7, df=3.0, sigma=0.011) -> pd.Series:
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(seed)
    raw = rng.standard_t(df, n)
    return pd.Series(raw / np.sqrt(df / (df - 2)) * sigma, index=idx)


# --- Path simulation ---------------------------------------------------------------------

def test_simulate_returns_one_cumulative_return_per_path():
    rng = np.random.default_rng(0)
    out = simulate_paths(normal_returns(), paths=500, horizon_days=10,
                         draw_method="bootstrap", rng=rng)
    assert out.shape == (500,)


def test_multi_day_paths_sum_drawn_days_rather_than_scaling_one():
    """Log returns are additive across time, which is why the engine works in log space. A
    ten-day path must be ten drawn days added, not one day multiplied by sqrt(10)."""
    returns = pd.Series([0.01] * 100)  # every draw identical, so the sum is deterministic
    rng = np.random.default_rng(0)
    out = simulate_paths(returns, paths=50, horizon_days=10, draw_method="bootstrap", rng=rng)
    assert np.allclose(out, 0.10)
    assert not np.allclose(out, 0.01 * np.sqrt(10))


def test_bootstrap_only_ever_draws_values_that_actually_occurred():
    returns = pd.Series([-0.05, 0.0, 0.03])
    rng = np.random.default_rng(0)
    out = simulate_paths(returns, paths=2000, horizon_days=1, draw_method="bootstrap", rng=rng)
    assert set(np.round(out, 10)).issubset({-0.05, 0.0, 0.03})


def test_an_unknown_draw_method_is_refused():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="unknown draw method"):
        simulate_paths(normal_returns(), 100, 1, "cauchy", rng)


def test_the_t_draw_requires_its_degrees_of_freedom():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        simulate_paths(normal_returns(), 100, 1, "t", rng, degrees_of_freedom=None)


def test_the_t_draw_is_rescaled_to_unit_variance():
    """Without the rescale the simulated spread is counted twice and every VaR inflates."""
    returns = normal_returns(sigma=0.01)
    rng = np.random.default_rng(1)
    drawn = simulate_paths(returns, paths=200_000, horizon_days=1,
                           draw_method="t", rng=rng, degrees_of_freedom=6.0)
    assert float(np.std(drawn)) == pytest.approx(0.01, rel=0.10)


# --- The bootstrap's ceiling, and where it lifts ---------------------------------------------

def test_a_one_day_bootstrap_cannot_beat_the_worst_observed_day():
    """Resampling samples from what happened, so at one day it shares historical VaR's
    ceiling. Claiming otherwise would imply simulation had escaped the sample."""
    returns = fat_tailed_returns()
    r = monte_carlo_var(returns, 0.99, horizon_days=1, draw_method="bootstrap", paths=20_000)
    assert not r.exceeds_history
    assert any("shares historical VaR's ceiling" in n for n in r.notes)


def test_a_multi_day_bootstrap_can_be_far_worse_than_any_single_day():
    """This is what resampling adds over Phase 4: ten independently drawn bad days compound
    into a path worse than anything in the record."""
    returns = fat_tailed_returns()
    one = monte_carlo_var(returns, 0.99, horizon_days=1, draw_method="bootstrap", paths=20_000)
    ten = monte_carlo_var(returns, 0.99, horizon_days=10, draw_method="bootstrap", paths=20_000)
    assert ten.worst_simulated_loss > one.worst_observed_loss
    assert any("far worse than any single historical day" in n for n in ten.notes)


def test_a_parametric_draw_can_exceed_history_even_at_one_day():
    """The trade against the bootstrap: a fitted distribution has no sample ceiling."""
    returns = fat_tailed_returns()
    r = monte_carlo_var(returns, 0.99, horizon_days=1, draw_method="t", paths=50_000)
    assert r.exceeds_history


# --- Fat tails must survive the simulation ------------------------------------------------------

def test_bootstrapping_fat_tailed_data_keeps_the_fat_tail():
    """A simulation that quietly normalises the data would defeat its own purpose."""
    returns = fat_tailed_returns()
    empirical = historical_var(returns, 0.99).var_return
    simulated = monte_carlo_var(returns, 0.99, draw_method="bootstrap", paths=50_000).var_return
    assert simulated == pytest.approx(empirical, rel=0.10)


def test_a_normal_draw_understates_the_tail_of_fat_tailed_data_and_says_so():
    returns = fat_tailed_returns()
    normal = monte_carlo_var(returns, 0.99, draw_method="normal", paths=50_000)
    boot = monte_carlo_var(returns, 0.99, draw_method="bootstrap", paths=50_000)
    assert normal.var_return < boot.var_return
    assert any("the data itself rejects" in w for w in normal.warnings)


def test_a_normal_draw_on_genuinely_normal_data_raises_no_such_warning():
    r = monte_carlo_var(normal_returns(), 0.99, draw_method="normal", paths=20_000)
    assert not any("the data itself rejects" in w for w in r.warnings)


# --- Simulation error, reproducibility and convergence --------------------------------------------

def test_the_same_seed_reproduces_the_same_answer():
    a = monte_carlo_var(normal_returns(), 0.95, seed=99, paths=5_000)
    b = monte_carlo_var(normal_returns(), 0.95, seed=99, paths=5_000)
    assert a.var_return == b.var_return


def test_different_seeds_agree_within_the_reported_standard_error():
    """The reported error is what makes 'run more paths' a decision rather than a guess, so
    it has to actually describe the spread across seeds."""
    runs = [monte_carlo_var(normal_returns(), 0.95, seed=s, paths=20_000).var_return
            for s in (1, 2, 3, 4, 5)]
    reported = monte_carlo_var(normal_returns(), 0.95, seed=1, paths=20_000).standard_error
    assert float(np.std(runs, ddof=1)) < reported * 6
    assert max(runs) - min(runs) < 0.002


def test_more_paths_reduce_the_simulation_error():
    small = monte_carlo_var(normal_returns(), 0.99, paths=1_000).standard_error
    large = monte_carlo_var(normal_returns(), 0.99, paths=50_000).standard_error
    assert large < small


def test_too_few_simulated_tail_points_is_warned_about():
    r = monte_carlo_var(normal_returns(), 0.99, paths=1_000)
    assert r.simulated_tail_observations < MIN_SIMULATED_TAIL
    assert any("Run more paths" in w for w in r.warnings)


def test_convergence_path_settles_as_paths_increase():
    frame = convergence_path(normal_returns(), 0.95)
    assert list(frame["paths"]) == sorted(frame["paths"])
    early = frame["change_from_previous"].iloc[1:3].mean()
    late = frame["change_from_previous"].iloc[-2:].mean()
    assert late < early


# --- Horizon and validation -------------------------------------------------------------------

def test_a_ten_day_var_exceeds_a_one_day_var_and_warns_about_clustering():
    returns = normal_returns()
    one = monte_carlo_var(returns, 0.95, horizon_days=1, paths=20_000)
    ten = monte_carlo_var(returns, 0.95, horizon_days=10, paths=20_000)
    assert ten.var_return > one.var_return
    assert any("clustering" in w for w in ten.warnings)


def test_too_few_paths_is_refused_rather_than_estimated():
    with pytest.raises(ValueError, match="at least 100 paths"):
        monte_carlo_var(normal_returns(), paths=50)


def test_an_impossible_confidence_level_is_refused():
    with pytest.raises(ValueError, match="confidence"):
        monte_carlo_var(normal_returns(), confidence=0.2)


def test_portfolio_value_scales_the_loss_linearly():
    r = monte_carlo_var(normal_returns(), 0.95, portfolio_value=500_000, paths=5_000)
    assert r.var_value == pytest.approx(r.var_return * 500_000)
