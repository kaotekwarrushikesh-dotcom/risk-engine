"""Tests for Phase 5: parametric VaR.

The failure modes worth pinning are the silent ones: a Student-t used without rescaling to
unit variance (which inflates every t-VaR), a mean scaled by sqrt(h) instead of h, and a
normal VaR presented without the normality test that would have rejected it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_engine.var_historical import historical_var
from risk_engine.var_parametric import (
    MAX_DEGREES_OF_FREEDOM,
    estimate_degrees_of_freedom,
    fit_distribution,
    normal_vs_historical_gap,
    parametric_var,
)


def normal_returns(n=2000, seed=7, mu=0.0004, sigma=0.011) -> pd.Series:
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(np.random.default_rng(seed).normal(mu, sigma, n), index=idx)


def fat_tailed_returns(n=2000, seed=7, df=3.0, sigma=0.011) -> pd.Series:
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(seed)
    raw = rng.standard_t(df, n)
    # Rescale to the target standard deviation so the two fixtures differ in shape, not spread.
    return pd.Series(raw / np.sqrt(df / (df - 2)) * sigma, index=idx)


# --- Distribution fitting and the normality test --------------------------------------------

def test_normal_data_is_not_rejected_as_normal():
    fit = fit_distribution(normal_returns())
    assert fit.is_normal
    assert not fit.has_fat_tails
    assert abs(fit.excess_kurtosis) < 0.5


def test_fat_tailed_data_is_rejected_as_normal():
    fit = fit_distribution(fat_tailed_returns())
    assert not fit.is_normal
    assert fit.has_fat_tails
    assert fit.excess_kurtosis > 1.0
    assert "understates the tail" in fit.verdict()


def test_fit_reports_the_moments_it_measured():
    returns = normal_returns(mu=0.001, sigma=0.02)
    fit = fit_distribution(returns)
    assert fit.mean == pytest.approx(float(returns.mean()))
    assert fit.std == pytest.approx(float(returns.std(ddof=1)))
    assert fit.n_observations == len(returns)


def test_too_little_data_to_judge_shape_is_refused():
    with pytest.raises(ValueError, match="eight observations"):
        fit_distribution(pd.Series([0.01, -0.02, 0.005]))


# --- Normal VaR against the closed form ------------------------------------------------------

def test_normal_var_matches_the_closed_form_by_hand():
    returns = normal_returns()
    r = parametric_var(returns, confidence=0.95, distribution="normal")
    mu, sigma = float(returns.mean()), float(returns.std(ddof=1))
    expected_quantile = mu + stats.norm.ppf(0.05) * sigma
    assert r.var_return == pytest.approx(-(np.exp(expected_quantile) - 1.0))


def test_a_higher_confidence_level_gives_a_larger_loss():
    returns = normal_returns()
    lo = parametric_var(returns, 0.95, distribution="normal").var_return
    hi = parametric_var(returns, 0.99, distribution="normal").var_return
    assert hi > lo


def test_normal_var_warns_when_its_own_assumption_is_rejected():
    """The assumption is checked rather than asserted, which is the entire point of running
    a normality test before reporting a normal VaR."""
    r = parametric_var(fat_tailed_returns(), 0.99, distribution="normal")
    assert not r.fit.is_normal
    assert any("understates the tail" in w for w in r.warnings)


def test_normal_var_on_genuinely_normal_data_raises_no_assumption_warning():
    r = parametric_var(normal_returns(), 0.99, distribution="normal")
    assert r.fit.is_normal
    assert not any("understates the tail" in w for w in r.warnings)


# --- The Student-t variant, and the rescaling that is easy to omit ---------------------------

def test_t_degrees_of_freedom_are_estimated_from_the_data():
    """Fat-tailed data must produce a low v; near-normal data a high one."""
    fat = estimate_degrees_of_freedom(fat_tailed_returns(df=3.0))
    thin = estimate_degrees_of_freedom(normal_returns())
    assert fat < 8
    assert thin > fat


def test_t_quantile_is_rescaled_to_unit_variance():
    """A standard t with v degrees of freedom has variance v/(v-2), not 1. Skipping the
    rescale double-counts the spread and inflates every t-VaR."""
    returns = fat_tailed_returns()
    r = parametric_var(returns, 0.99, distribution="t")
    v = r.degrees_of_freedom
    mu, sigma = float(returns.mean()), float(returns.std(ddof=1))

    scale = np.sqrt((v - 2.0) / v)
    expected = mu + stats.t.ppf(0.01, v) * scale * sigma
    assert r.var_return == pytest.approx(-(np.exp(expected) - 1.0))

    # And the unscaled version, the bug this guards against, would be materially larger.
    unscaled = mu + stats.t.ppf(0.01, v) * sigma
    assert -(np.exp(unscaled) - 1.0) > r.var_return


def test_t_var_exceeds_normal_var_in_the_tail_on_fat_tailed_data():
    """This is the whole reason the t variant exists."""
    returns = fat_tailed_returns()
    normal = parametric_var(returns, 0.99, distribution="normal").var_return
    student = parametric_var(returns, 0.99, distribution="t").var_return
    assert student > normal


def test_t_and_normal_converge_when_the_data_is_actually_normal():
    returns = normal_returns()
    normal = parametric_var(returns, 0.99, distribution="normal").var_return
    student = parametric_var(returns, 0.99, distribution="t").var_return
    assert student == pytest.approx(normal, rel=0.15)


def test_a_t_fit_that_is_indistinguishable_from_normal_says_so():
    # Perfectly normal data drives the fitted v to the ceiling.
    returns = pd.Series(stats.norm.rvs(0, 0.01, size=5000, random_state=1))
    r = parametric_var(returns, 0.99, distribution="t")
    if r.degrees_of_freedom >= MAX_DEGREES_OF_FREEDOM:
        assert any("indistinguishable from a normal" in w for w in r.warnings)


# --- Horizon scaling -------------------------------------------------------------------------

def test_mean_scales_linearly_and_volatility_by_its_square_root():
    """Scaling both by sqrt(h) is a common and quietly wrong shortcut."""
    returns = normal_returns(mu=0.001, sigma=0.02)
    r = parametric_var(returns, 0.95, horizon_days=10, distribution="normal")

    mu, sigma = float(returns.mean()), float(returns.std(ddof=1))
    expected_quantile = mu * 10 + stats.norm.ppf(0.05) * sigma * np.sqrt(10)
    assert r.var_return == pytest.approx(-(np.exp(expected_quantile) - 1.0))

    wrong = mu * np.sqrt(10) + stats.norm.ppf(0.05) * sigma * np.sqrt(10)
    assert r.var_return != pytest.approx(-(np.exp(wrong) - 1.0))


def test_a_ten_day_var_is_larger_than_a_one_day_var():
    returns = normal_returns()
    one = parametric_var(returns, 0.95, horizon_days=1).var_return
    ten = parametric_var(returns, 0.95, horizon_days=10).var_return
    assert ten > one


# --- Comparison against the historical method -------------------------------------------------

def test_the_gap_against_historical_var_reads_the_fat_tail():
    returns = fat_tailed_returns()
    p = parametric_var(returns, 0.99, distribution="normal").var_return
    h = historical_var(returns, 0.99).var_return
    gap = normal_vs_historical_gap(p, h)
    assert gap["gap"] > 0
    assert gap["ratio"] > 1
    assert "fat tail" in gap["reading"]


def test_a_negative_gap_is_flagged_as_a_thin_sample_not_a_clean_bill_of_health():
    """A normal VaR larger than the historical one usually means the sample has not held a
    bad enough day yet, which is Phase 4's blind spot rather than a well-behaved market."""
    gap = normal_vs_historical_gap(0.05, 0.03)
    assert gap["gap"] < 0
    assert "has not yet contained a loss bad enough" in gap["reading"]


# --- Input validation ---------------------------------------------------------------------------

def test_an_unknown_distribution_is_refused():
    with pytest.raises(ValueError, match="unknown distribution"):
        parametric_var(normal_returns(), distribution="cauchy")


def test_an_impossible_confidence_level_is_refused():
    with pytest.raises(ValueError, match="confidence"):
        parametric_var(normal_returns(), confidence=1.2)


def test_a_horizon_below_one_day_is_refused():
    with pytest.raises(ValueError, match="horizon_days"):
        parametric_var(normal_returns(), horizon_days=0)


def test_portfolio_value_scales_the_loss_linearly():
    returns = normal_returns()
    r = parametric_var(returns, 0.95, portfolio_value=2_000_000)
    assert r.var_value == pytest.approx(r.var_return * 2_000_000)
