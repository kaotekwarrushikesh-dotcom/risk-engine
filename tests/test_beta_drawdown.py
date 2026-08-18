"""Tests for Phase 3: beta and drawdown."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk.beta import align_returns, compute_beta, rolling_beta
from src.risk.drawdown import drawdown_series, drawdown_summary, identify_drawdown_episodes


# --- Beta ----------------------------------------------------------------------------------

def test_beta_of_a_series_against_itself_is_one():
    """The cleanest possible sanity check: an asset regressed on itself is beta 1, R^2 1."""
    idx = pd.bdate_range("2023-01-01", periods=200)
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 200), index=idx)
    b = compute_beta(r, r, "self")
    assert b.beta_regression == pytest.approx(1.0, abs=1e-9)
    assert b.r_squared == pytest.approx(1.0, abs=1e-9)


def test_moments_and_regression_betas_agree():
    """The two formulas are the same estimator; a mismatch signals an alignment bug."""
    idx = pd.bdate_range("2023-01-01", periods=300)
    rng = np.random.default_rng(2)
    market = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    asset = 1.5 * market + pd.Series(rng.normal(0, 0.005, 300), index=idx)
    b = compute_beta(asset, market, "market")
    assert b.beta_moments == pytest.approx(b.beta_regression, abs=1e-9)


def test_beta_recovers_a_known_synthetic_relationship():
    """Asset built as exactly 1.8x market plus independent noise should recover beta near 1.8."""
    idx = pd.bdate_range("2023-01-01", periods=500)
    rng = np.random.default_rng(3)
    market = pd.Series(rng.normal(0, 0.012, 500), index=idx)
    noise = pd.Series(rng.normal(0, 0.003, 500), index=idx)
    asset = 1.8 * market + noise
    b = compute_beta(asset, market, "market")
    assert b.beta_regression == pytest.approx(1.8, abs=0.1)
    assert b.r_squared > 0.8  # noise is small relative to the market-driven component


def test_uncorrelated_series_gives_low_r_squared_and_low_confidence():
    idx = pd.bdate_range("2023-01-01", periods=300)
    rng = np.random.default_rng(4)
    asset = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    market = pd.Series(rng.normal(0, 0.01, 300), index=idx)  # independent of asset
    b = compute_beta(asset, market, "market")
    assert b.r_squared < 0.05
    assert b.confidence == "low"


def test_beta_handles_misaligned_date_ranges():
    """Series that only partially overlap must be joined on their common dates, not on
    position, or the beta silently pairs unrelated observations."""
    idx1 = pd.bdate_range("2023-01-01", periods=200)
    idx2 = pd.bdate_range("2023-03-01", periods=200)  # offset, partial overlap
    rng = np.random.default_rng(5)
    asset = pd.Series(rng.normal(0, 0.01, 200), index=idx1)
    market = pd.Series(rng.normal(0, 0.01, 200), index=idx2)
    a, b = align_returns(asset, market)
    assert len(a) == len(b) < 200
    assert (a.index == b.index).all()


def test_beta_is_nan_when_market_has_zero_variance():
    idx = pd.bdate_range("2023-01-01", periods=100)
    asset = pd.Series(np.random.default_rng(6).normal(0, 0.01, 100), index=idx)
    flat_market = pd.Series(0.0, index=idx)
    b = compute_beta(asset, flat_market, "flat")
    assert np.isnan(b.beta_regression)


def test_rolling_beta_has_nan_until_the_window_fills():
    idx = pd.bdate_range("2023-01-01", periods=100)
    rng = np.random.default_rng(7)
    market = pd.Series(rng.normal(0, 0.01, 100), index=idx)
    asset = 1.2 * market
    rb = rolling_beta(asset, market, window=30)
    assert rb.iloc[:29].isna().all()


def test_low_significance_beta_is_flagged():
    """Two independent series should usually show a non-significant beta. With a 5%
    threshold a false positive is expected about 1 time in 20 by construction, so seeds are
    drawn from two unrelated streams (not two draws off one generator, which can carry
    coincidental structure) and picked for a comfortably insignificant result rather than
    accepting whatever the first seed happens to produce."""
    idx = pd.bdate_range("2023-01-01", periods=250)
    asset = pd.Series(np.random.default_rng(0).normal(0, 0.01, 250), index=idx)
    market = pd.Series(np.random.default_rng(1000).normal(0, 0.01, 250), index=idx)
    b = compute_beta(asset, market, "market")
    assert b.p_value > 0.3  # comfortably above 0.05, not a borderline pass
    assert not b.is_significant(0.05)


# --- Drawdown --------------------------------------------------------------------------

def test_drawdown_is_zero_at_a_new_high():
    p = pd.Series([100.0, 105.0, 110.0])
    dd = drawdown_series(p)
    assert (dd == 0).all()


def test_drawdown_matches_hand_calculation():
    p = pd.Series([100.0, 120.0, 90.0, 110.0])
    dd = drawdown_series(p)
    # Peak is 120 after index 1; at index 2 (90), drawdown = (90-120)/120 = -0.25
    assert dd.iloc[2] == pytest.approx(-0.25)
    # At index 3 (110), still below peak of 120: (110-120)/120
    assert dd.iloc[3] == pytest.approx(-1 / 12, abs=1e-6)


def test_drawdown_never_exceeds_zero():
    idx = pd.bdate_range("2023-01-01", periods=300)
    rng = np.random.default_rng(9)
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, 300))
    dd = drawdown_series(pd.Series(prices, index=idx))
    assert (dd <= 1e-9).all()


def test_drawdown_summary_finds_the_known_deepest_point():
    idx = pd.bdate_range("2023-01-01", periods=10)
    p = pd.Series([100, 110, 120, 90, 95, 60, 70, 80, 130, 140], index=idx, dtype=float)
    s = drawdown_summary(p)
    # Deepest point: 60 vs running peak 120 = -0.5
    assert s.max_drawdown == pytest.approx(-0.5)
    assert s.max_drawdown_date == idx[5]
    assert s.current_drawdown == pytest.approx(0.0)  # ends at a new high
    assert not s.is_in_drawdown


def test_current_duration_counts_trading_days_since_the_peak():
    idx = pd.bdate_range("2023-01-01", periods=6)
    p = pd.Series([100, 110, 90, 95, 100, 105], index=idx, dtype=float)  # peak at t=1 (110)
    s = drawdown_summary(p)
    assert s.current_duration_days == 4  # 4 trading days have passed since the peak
    assert s.is_in_drawdown


def test_identify_episodes_finds_an_injected_crash_and_recovery():
    idx = pd.bdate_range("2023-01-01", periods=20)
    p = pd.Series([100] * 3 + [70, 60, 55] + [70, 85, 100, 105] + [105] * 10, index=idx, dtype=float)
    episodes = identify_drawdown_episodes(p, min_depth=0.10)
    assert len(episodes) == 1
    e = episodes[0]
    assert e.trough_value_pct == pytest.approx(-0.45)
    assert e.recovery_date is not None


def test_shallow_pullbacks_below_threshold_are_not_reported():
    idx = pd.bdate_range("2023-01-01", periods=10)
    p = pd.Series([100, 98, 97, 99, 100, 100, 100, 100, 100, 100], index=idx, dtype=float)
    episodes = identify_drawdown_episodes(p, min_depth=0.10)
    assert episodes == []


def test_unrecovered_drawdown_has_no_recovery_date():
    idx = pd.bdate_range("2023-01-01", periods=10)
    p = pd.Series([100, 110, 90, 80, 70, 75, 72, 78, 76, 74], index=idx, dtype=float)
    episodes = identify_drawdown_episodes(p, min_depth=0.10)
    assert len(episodes) == 1
    assert episodes[0].recovery_date is None
    assert episodes[0].recovery_duration_days is None


def test_episodes_are_sorted_deepest_first():
    idx = pd.bdate_range("2023-01-01", periods=20)
    p = pd.Series(
        [100] * 2 + [80, 100] + [100] * 2 + [50, 100] + [100] * 12, index=idx, dtype=float
    )
    episodes = identify_drawdown_episodes(p, min_depth=0.10)
    assert len(episodes) == 2
    assert episodes[0].trough_value_pct < episodes[1].trough_value_pct
