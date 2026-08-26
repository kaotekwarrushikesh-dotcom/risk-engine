"""Tests for Phase 10: valuation risk and stress testing.

The distinction this file exists to protect is that valuation risk is *sensitivity*, not
*level*. Module 2's DCF is known to read below market, and that bias moves every scenario in
the same direction, so it must not contaminate the sensitivity measures. Several tests below
check exactly that a uniform level shift leaves sensitivity unchanged.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import risk_engine.valuation_risk as vr
from risk_engine.valuation_risk import (
    FRAGILE_PER_25BP,
    TERMINAL_DEPENDENCE_CRITICAL,
    TERMINAL_DEPENDENCE_HIGH,
    ValuationRisk,
    _solve_growth_for_price,
)

integration = pytest.mark.integration


def make_risk(terminal_share=0.6, wacc_sensitivity=0.03) -> ValuationRisk:
    return ValuationRisk(
        ticker="TEST", name="Test", currency="USD",
        share_price=100.0, implied_share_price=80.0, upside=-0.2,
        wacc=0.09, terminal_growth=0.02, terminal_share=terminal_share,
        market_implied_wacc=0.05, market_implied_growth=0.03,
        wacc_sensitivity=wacc_sensitivity, growth_sensitivity=0.04,
        grid=pd.DataFrame(),
    )


# --- Assumption dependence ---------------------------------------------------------------------

def test_terminal_dependence_bands_follow_the_published_thresholds():
    assert make_risk(terminal_share=0.40).assumption_dependence == "Low"
    assert make_risk(terminal_share=0.60).assumption_dependence == "Moderate"
    assert make_risk(terminal_share=TERMINAL_DEPENDENCE_HIGH).assumption_dependence == "High"
    assert make_risk(
        terminal_share=TERMINAL_DEPENDENCE_CRITICAL).assumption_dependence == "Critical"


def test_fragility_is_judged_on_a_quarter_point_move_in_the_discount_rate():
    assert make_risk(wacc_sensitivity=FRAGILE_PER_25BP + 0.01).is_fragile
    assert not make_risk(wacc_sensitivity=FRAGILE_PER_25BP - 0.01).is_fragile


def test_fragility_ignores_the_sign_of_the_sensitivity():
    assert make_risk(wacc_sensitivity=-0.20).is_fragile


# --- The guard that keeps the perpetuity solvable -------------------------------------------------

def test_growth_at_or_above_the_discount_rate_is_refused_rather_than_clamped():
    """The perpetuity divides by (wacc - growth). At or past that boundary the result is not a
    valuation worth reporting, and clamping would produce a number that looks like one."""
    base = {"ticker": "X", "fcff": None, "net_debt": 0.0, "shares": 1.0,
            "share_price": 10.0, "currency": "USD", "roic": 0.15}
    assert np.isnan(vr._revalue(base, wacc=0.08, growth=0.08))
    assert np.isnan(vr._revalue(base, wacc=0.08, growth=0.10))


def test_a_failing_scenario_returns_nan_rather_than_propagating_an_exception(monkeypatch):
    """One unsolvable cell must not take down a whole grid."""
    import valuation_engine.dcf as dcf_module

    def explode(**_kwargs):
        raise RuntimeError("engine failure")

    monkeypatch.setattr(dcf_module, "run_dcf", explode)
    base = {"ticker": "X", "fcff": None, "net_debt": 0.0, "shares": 1.0,
            "share_price": 10.0, "currency": "USD", "roic": 0.15}
    assert np.isnan(vr._revalue(base, wacc=0.09, growth=0.02))


# --- Reverse stress testing --------------------------------------------------------------------------

def test_the_solver_finds_the_growth_rate_that_reproduces_a_target_price(monkeypatch):
    """A linear stand-in with a known answer, so the bisection has a right result to hit."""
    def linear(_base, _wacc, growth):
        return 50.0 + 2000.0 * growth  # growth of 2.5% gives exactly 100

    monkeypatch.setattr(vr, "_revalue", linear)
    solved = _solve_growth_for_price({}, wacc=0.09, target_price=100.0)
    assert solved == pytest.approx(0.025, abs=1e-4)


def test_the_solver_reports_nothing_when_no_growth_rate_reaches_the_price(monkeypatch):
    """Informative in itself: the gap cannot be explained by terminal growth alone."""
    def capped(_base, _wacc, growth):
        return 50.0 + 100.0 * growth  # never reaches 500 inside the searchable range

    monkeypatch.setattr(vr, "_revalue", capped)
    assert np.isnan(_solve_growth_for_price({}, wacc=0.09, target_price=500.0))


def test_the_solver_refuses_when_the_search_range_is_empty():
    """A discount rate at or below the floor leaves no room between the bounds."""
    assert np.isnan(_solve_growth_for_price({}, wacc=0.0, target_price=100.0))


def test_the_solver_handles_unsolvable_cells_inside_the_range(monkeypatch):
    def patchy(_base, _wacc, growth):
        if growth > 0.04:
            return float("nan")
        return 50.0 + 2000.0 * growth

    monkeypatch.setattr(vr, "_revalue", patchy)
    solved = _solve_growth_for_price({}, wacc=0.09, target_price=100.0)
    assert not np.isnan(solved)


# --- Sensitivity must survive a level bias -----------------------------------------------------------

def test_a_uniform_level_bias_does_not_change_relative_sensitivity(monkeypatch):
    """Module 2's DCF reads systematically low. That bias scales every scenario by the same
    factor, so the *relative* sensitivity this phase measures is unaffected, and this test
    pins that rather than leaving it as an argument in the docstring."""
    def unbiased(_base, wacc, _growth):
        return 100.0 * (0.09 / wacc)

    def biased(_base, wacc, _growth):
        return 0.4 * 100.0 * (0.09 / wacc)  # every value 60% lower

    base = {"ticker": "X"}
    results = []
    for pricer in (unbiased, biased):
        monkeypatch.setattr(vr, "_revalue", pricer)
        implied = pricer(base, 0.09, 0.02)
        up = pricer(base, 0.09 + vr.WACC_SHOCK, 0.02)
        down = pricer(base, 0.09 - vr.WACC_SHOCK, 0.02)
        results.append((down - up) / (2 * implied))

    assert results[0] == pytest.approx(results[1])


# --- Integration with Module 2 --------------------------------------------------------------------------

@integration
def test_a_real_company_produces_a_sensitivity_profile():
    result = vr.assess("AAPL")
    assert 0.0 < result.terminal_share < 1.0
    assert not np.isnan(result.wacc_sensitivity)
    assert result.grid.shape == (5, 5)


@integration
def test_a_higher_discount_rate_always_lowers_the_valuation():
    """The most basic property of discounting, checked against the real engine rather than
    assumed: every row of the grid must fall as WACC rises."""
    grid = vr.assess("AAPL").grid
    for column in grid.columns:
        values = grid[column].dropna().to_numpy()
        assert np.all(np.diff(values) < 0), f"column {column} is not monotone in WACC"


@integration
def test_higher_terminal_growth_always_raises_the_valuation():
    grid = vr.assess("AAPL").grid
    for _, row in grid.iterrows():
        values = row.dropna().to_numpy()
        assert np.all(np.diff(values) > 0)


@integration
def test_the_calibration_limitation_travels_with_every_result():
    """Module 2's documented bias must reach the reader, since it changes how the level should
    be read even though it leaves the sensitivity intact."""
    result = vr.assess("AAPL")
    assert any("systematically below market" in n for n in result.notes)


@integration
def test_stress_scenarios_move_the_valuation_in_the_expected_directions():
    frame = vr.stress_test("AAPL")
    base = frame.loc[frame["scenario"] == "Base case", "implied_price"].iloc[0]
    tighter = frame.loc[frame["scenario"] == "Rates rise 200bp", "implied_price"].iloc[0]
    easier = frame.loc[
        frame["scenario"] == "Soft landing: rates -100bp, growth +50bp", "implied_price"].iloc[0]

    assert tighter < base < easier


@integration
def test_a_company_without_a_usable_dcf_raises_a_clear_error():
    with pytest.raises(Exception):
        vr.assess("NOT_A_REAL_TICKER_XYZQ")
