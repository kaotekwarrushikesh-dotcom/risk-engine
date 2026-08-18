"""Tests for Phase 2: returns and volatility."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk.returns import (
    annualise_return,
    compute_returns,
    cumulative_return,
    log_returns,
    simple_returns,
    to_price_index,
)
from src.risk.volatility import (
    annualise_volatility,
    historical_volatility,
    identify_volatility_spikes,
    rolling_volatility,
    volatility_regime,
)


def make_prices(n=300, seed=1, drift=0.0002, vol=0.015) -> pd.Series:
    idx = pd.bdate_range("2023-01-01", periods=n)
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(drift, vol, n)
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.Series(prices, index=idx)


# --- Returns -----------------------------------------------------------------------------

def test_simple_return_matches_hand_calculation():
    p = pd.Series([100.0, 110.0, 99.0])
    r = simple_returns(p)
    assert r.iloc[0] == pytest.approx(0.10)
    assert r.iloc[1] == pytest.approx(-0.10)


def test_log_return_matches_hand_calculation():
    p = pd.Series([100.0, 110.0, 99.0])
    r = log_returns(p)
    assert r.iloc[0] == pytest.approx(np.log(1.10))
    assert r.iloc[1] == pytest.approx(np.log(0.9))


def test_log_and_simple_agree_for_small_moves():
    p = pd.Series([100.0, 100.1, 100.05])
    assert simple_returns(p).iloc[0] == pytest.approx(log_returns(p).iloc[0], abs=1e-4)


def test_log_and_simple_diverge_for_large_moves():
    p = pd.Series([100.0, 200.0])
    simple = simple_returns(p).iloc[0]
    log = log_returns(p).iloc[0]
    assert simple == pytest.approx(1.0)
    assert log == pytest.approx(np.log(2.0))
    assert abs(simple - log) > 0.1


def test_compute_returns_dispatches_correctly():
    p = make_prices(50)
    assert compute_returns(p, "simple").equals(simple_returns(p))
    assert compute_returns(p, "log").equals(log_returns(p))


def test_compute_returns_rejects_unknown_method():
    with pytest.raises(ValueError):
        compute_returns(make_prices(50), "made_up_method")


def test_cumulative_log_return_converts_to_the_same_simple_return():
    """exp(sum of log returns) - 1 must equal the simple cumulative return: both describe
    the same total price move in different units."""
    p = make_prices(200)
    log_cum = cumulative_return(log_returns(p), "log")
    simple_cum = cumulative_return(simple_returns(p), "simple")
    assert np.exp(log_cum) - 1.0 == pytest.approx(simple_cum, abs=1e-6)


def test_annualise_return_log_is_linear_simple_is_geometric():
    assert annualise_return(0.001, 252, "log") == pytest.approx(0.252)
    assert annualise_return(0.001, 252, "simple") == pytest.approx(1.001**252 - 1)
    # For a positive mean return, geometric compounding of simple returns exceeds the
    # linear scaling used for log returns.
    assert annualise_return(0.001, 252, "simple") > annualise_return(0.001, 252, "log")


def test_to_price_index_round_trips_through_log_returns():
    p = make_prices(100)
    idx = to_price_index(log_returns(p), "log", base=p.iloc[0])
    assert idx.iloc[-1] == pytest.approx(p.iloc[-1], rel=1e-6)


# --- Volatility --------------------------------------------------------------------------

def test_annualise_volatility_scales_by_sqrt_time():
    assert annualise_volatility(0.01, 252) == pytest.approx(0.01 * np.sqrt(252))


def test_historical_volatility_horizons_are_consistently_scaled():
    r = log_returns(make_prices(300))
    v = historical_volatility(r, window=100)
    assert v.weekly == pytest.approx(v.daily * np.sqrt(5))
    assert v.monthly == pytest.approx(v.daily * np.sqrt(21))
    assert v.annualised == pytest.approx(v.daily * np.sqrt(252))
    assert v.n_observations == 100


def test_historical_volatility_matches_known_input_std():
    """A series with a known, injected standard deviation should recover it, not merely
    something in the right ballpark."""
    idx = pd.bdate_range("2023-01-01", periods=252)
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0, 0.02, 252), index=idx)
    v = historical_volatility(r)
    assert v.daily == pytest.approx(r.std(ddof=1))


def test_historical_volatility_handles_a_single_observation():
    v = historical_volatility(pd.Series([0.01]))
    assert np.isnan(v.daily)


def test_rolling_volatility_has_nan_until_the_window_fills():
    r = log_returns(make_prices(100))
    rv = rolling_volatility(r, window=20)
    assert rv.iloc[:19].isna().all()
    assert rv.iloc[19:].notna().all()


def test_rolling_volatility_reacts_to_a_regime_change():
    """A calm-then-turbulent series must show materially higher rolling volatility once the
    turbulent window has fully entered the rolling calculation."""
    calm = pd.Series(np.random.default_rng(1).normal(0, 0.005, 100))
    turbulent = pd.Series(np.random.default_rng(2).normal(0, 0.05, 100))
    combined = pd.concat([calm, turbulent], ignore_index=True)
    rv = rolling_volatility(combined, window=20, annualise=False)
    assert rv.iloc[-1] > rv.iloc[95] * 3


def test_volatility_regime_classification():
    assert volatility_regime(0.40, 0.20) == "elevated"
    assert volatility_regime(0.10, 0.20) == "subdued"
    assert volatility_regime(0.20, 0.20) == "normal"
    assert volatility_regime(float("nan"), 0.20) == "unknown"


def test_identify_volatility_spikes_finds_an_injected_episode():
    idx = pd.bdate_range("2023-01-01", periods=200)
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 0.01, 200), index=idx)
    r.iloc[100:130] = rng.normal(0, 0.06, 30)  # a genuine stress episode
    rv = rolling_volatility(r, window=10, annualise=False)

    spikes = identify_volatility_spikes(rv, threshold_multiple=1.5)
    assert len(spikes) >= 1
    assert spikes["peak_volatility"].max() > rv.median() * 1.5


def test_identify_volatility_spikes_on_an_empty_series_returns_empty_frame():
    spikes = identify_volatility_spikes(pd.Series(dtype=float))
    assert spikes.empty
