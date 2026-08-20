"""Tests for Phase 8: GARCH(1,1) conditional volatility.

The properties worth pinning are the ones separating a fitted model from a trusted one: that
ARCH effects are tested before the model is fitted rather than assumed, that the residual
diagnostic actually detects a bad fit, that persistence is interpreted rather than reported,
and that the model-implied long-run level is suppressed when it is numerically meaningless.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_engine.garch import (
    UNRELIABLE_LONG_RUN_PERSISTENCE,
    arch_lm_test,
    conditional_var,
    fit_garch,
    forecast_volatility,
)


def constant_volatility_returns(n=1500, seed=5, sigma=0.01) -> pd.Series:
    """No clustering: variance is genuinely constant, so there is nothing for GARCH to model."""
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(np.random.default_rng(seed).normal(0.0003, sigma, n), index=idx)


def garch_returns(n=2000, seed=11, omega=0.02, alpha=0.10, beta=0.87) -> pd.Series:
    """Returns simulated from a known GARCH process, so the fit has a right answer to find."""
    idx = pd.bdate_range("2016-01-01", periods=n)
    rng = np.random.default_rng(seed)
    variance = omega / (1 - alpha - beta)
    out = np.zeros(n)
    for i in range(n):
        shock = rng.normal(0, np.sqrt(variance))
        out[i] = shock
        variance = omega + alpha * shock**2 + beta * variance
    return pd.Series(out / 100.0, index=idx)


# --- The ARCH-LM test, which justifies the model before it is fitted --------------------------

def test_arch_test_detects_clustering_where_it_exists():
    residuals = (garch_returns() * 100).to_numpy()
    result = arch_lm_test(residuals - residuals.mean())
    assert result.has_arch_effects
    assert result.p_value < 0.05
    assert "clustering for a GARCH model to capture" in result.reading()


def test_arch_test_finds_nothing_in_constant_volatility_data():
    residuals = (constant_volatility_returns() * 100).to_numpy()
    result = arch_lm_test(residuals - residuals.mean())
    assert not result.has_arch_effects
    assert "would be fitting noise" in result.reading()


def test_arch_test_refuses_a_sample_too_short_for_its_lags():
    with pytest.raises(ValueError, match="ARCH-LM"):
        arch_lm_test(np.arange(5.0), lags=10)


# --- Fitting, and recovering a known process ---------------------------------------------------

def test_the_fit_recovers_parameters_of_a_simulated_process():
    """A model that cannot recover a process it was handed is not measuring anything."""
    fit = fit_garch(garch_returns(omega=0.02, alpha=0.10, beta=0.87), distribution="normal")
    assert fit.converged
    assert fit.alpha == pytest.approx(0.10, abs=0.06)
    assert fit.beta == pytest.approx(0.87, abs=0.08)
    assert fit.persistence == pytest.approx(0.97, abs=0.03)


def test_a_good_fit_leaves_no_arch_effects_in_its_residuals():
    """If the clustering is still there afterwards, the model did not capture it."""
    fit = fit_garch(garch_returns())
    assert fit.arch_test_before.has_arch_effects
    assert fit.residuals_are_clean
    assert not any("ARCH effects remain" in w for w in fit.warnings)


def test_fitting_data_without_arch_effects_says_the_model_is_unnecessary():
    fit = fit_garch(constant_volatility_returns())
    assert not fit.arch_test_before.has_arch_effects
    assert any("fitting noise" in w for w in fit.warnings)


def test_too_little_data_is_refused_rather_than_fitted():
    with pytest.raises(ValueError, match="at least 250 observations"):
        fit_garch(garch_returns(n=100))


def test_an_unknown_error_distribution_is_refused():
    with pytest.raises(ValueError, match="unknown distribution"):
        fit_garch(garch_returns(), distribution="cauchy")


def test_the_student_t_variant_fits_a_tail_parameter():
    fit = fit_garch(garch_returns(), distribution="t")
    assert fit.nu is not None
    assert fit.nu > 2.0
    assert any("says nothing about the shape of" in n for n in fit.notes)


# --- Persistence, half-life and the long-run level -----------------------------------------------

def test_persistence_is_the_sum_of_alpha_and_beta():
    fit = fit_garch(garch_returns())
    assert fit.persistence == pytest.approx(fit.alpha + fit.beta)


def test_half_life_falls_as_persistence_falls():
    """Persistence is the number to read; half-life is what makes it concrete."""
    # omega is raised for the fast-decaying process so both fixtures sit at a comparable
    # variance level; otherwise low persistence alone shrinks the series enough that the
    # optimiser complains about scale rather than about anything meaningful.
    slow = fit_garch(garch_returns(omega=0.02, alpha=0.05, beta=0.94))
    fast = fit_garch(garch_returns(omega=0.30, alpha=0.15, beta=0.60))
    assert slow.half_life_days > fast.half_life_days


def test_a_stationary_fit_has_a_finite_long_run_level():
    fit = fit_garch(garch_returns(alpha=0.10, beta=0.70))
    assert fit.is_stationary
    assert np.isfinite(fit.long_run_variance)
    assert fit.long_run_is_reliable


def test_a_near_unit_root_fit_suppresses_its_long_run_level():
    """omega / (1 - persistence) divides by a number approaching zero, so near the boundary
    the implied level is numerically explosive and must not be quoted as a measurement."""
    fit = fit_garch(garch_returns(omega=0.005, alpha=0.09, beta=0.905))
    if fit.persistence > UNRELIABLE_LONG_RUN_PERSISTENCE:
        assert not fit.long_run_is_reliable
        assert any("should not be quoted" in w for w in fit.warnings)
        assert "unreliable at this persistence" in fit.summary()


def test_realised_volatility_is_always_available_as_a_fallback_benchmark():
    """A measurement rather than an extrapolation, so it stays usable when the model-implied
    long-run level does not."""
    fit = fit_garch(garch_returns())
    assert 0.0 < fit.realised_annualised_volatility < 2.0


# --- Forecasting -----------------------------------------------------------------------------------

def test_the_forecast_mean_reverts_toward_the_long_run_level():
    """The behaviour square-root-of-time scaling cannot produce: it projects today's
    volatility unchanged forever."""
    returns = garch_returns(alpha=0.10, beta=0.70)
    fit = fit_garch(returns)
    forecast = forecast_volatility(fit, horizon_days=60)

    start = float(forecast.annualised_volatility.iloc[0])
    end = float(forecast.annualised_volatility.iloc[-1])
    long_run = fit.long_run_annualised_volatility
    assert abs(end - long_run) < abs(start - long_run)


def test_the_forecast_moves_toward_the_long_run_from_either_side():
    fit = fit_garch(garch_returns(alpha=0.10, beta=0.70))
    forecast = forecast_volatility(fit, horizon_days=120)
    path = forecast.annualised_volatility.to_numpy()
    long_run = fit.long_run_annualised_volatility
    # Monotone approach: every step is at least as close as the previous one.
    distances = np.abs(path - long_run)
    assert np.all(np.diff(distances) <= 1e-9)


def test_cumulative_volatility_sums_variances_rather_than_scaling_one_day():
    """The terms being summed are unequal, which is exactly what sqrt-of-time cannot show."""
    fit = fit_garch(garch_returns())
    forecast = forecast_volatility(fit, horizon_days=10)
    naive = float(forecast.daily_volatility.iloc[0]) * np.sqrt(10)
    assert forecast.cumulative_volatility != pytest.approx(naive)
    assert any("cannot represent" in n for n in forecast.notes)


def test_a_horizon_below_one_day_is_refused():
    with pytest.raises(ValueError, match="horizon_days"):
        forecast_volatility(fit_garch(garch_returns()), horizon_days=0)


# --- Conditional VaR, the point of the phase --------------------------------------------------------

def test_conditional_var_responds_to_the_current_volatility_regime():
    """An unconditional VaR is the same number in a crisis and a calm week. This must not be."""
    returns = garch_returns()

    calm = returns.copy()
    calm.iloc[-30:] = calm.iloc[-30:] * 0.2
    stressed = returns.copy()
    stressed.iloc[-30:] = stressed.iloc[-30:] * 4.0

    calm_var = conditional_var(fit_garch(calm), 0.99)["var_return"]
    stressed_var = conditional_var(fit_garch(stressed), 0.99)["var_return"]
    assert stressed_var > calm_var


def test_the_regime_label_is_judged_against_realised_not_the_unstable_long_run():
    fit = fit_garch(garch_returns())
    result = conditional_var(fit, 0.95)
    assert result["regime"] in ("elevated", "subdued", "normal")
    assert result["realised_annualised_volatility"] == pytest.approx(
        fit.realised_annualised_volatility)


def test_a_higher_confidence_level_gives_a_larger_conditional_loss():
    fit = fit_garch(garch_returns())
    lo = conditional_var(fit, 0.95)["var_return"]
    hi = conditional_var(fit, 0.99)["var_return"]
    assert hi > lo


def test_a_longer_horizon_gives_a_larger_conditional_loss():
    fit = fit_garch(garch_returns())
    one = conditional_var(fit, 0.95, horizon_days=1)["var_return"]
    ten = conditional_var(fit, 0.95, horizon_days=10)["var_return"]
    assert ten > one


def test_portfolio_value_scales_the_conditional_loss_linearly():
    fit = fit_garch(garch_returns())
    r = conditional_var(fit, 0.95, portfolio_value=1_000_000)
    assert r["var_value"] == pytest.approx(r["var_return"] * 1_000_000)


def test_an_impossible_confidence_level_is_refused():
    with pytest.raises(ValueError, match="confidence"):
        conditional_var(fit_garch(garch_returns()), confidence=1.5)
